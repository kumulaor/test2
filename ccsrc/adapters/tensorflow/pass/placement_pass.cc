#include "adapters/tensorflow/pass/placement_pass.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "DistributedIR/graph.hpp"
#include "adapters/tensorflow/rpc/client.h"
#include "adapters/tensorflow/rpc/graph.pb.h"
#include "adapters/tensorflow/rpc/util.h"
#include "common/fmt.hpp"
#include "common/log.h"
#include "cost_graph/common.hpp"
#include "google/protobuf/text_format.h"
#include "json/json.h"
#include "policy/fd-dps/fddps_algorithm.h"
#include "policy/sgp/graphPartition.h"
#include "range/v3/all.hpp"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/tsl/platform/status.h"

namespace tensorflow {
std::map<std::string, framework::DataType> type_map = {{"int32", framework::DataType::I32},
                                                       {"float32", framework::DataType::F32},
                                                       {"float32_ref", framework::DataType::F32}};
void CalculateMemory(std::shared_ptr<framework::NodeBase> node) {
    // 输入数据大小
    std::int64_t input_total_size = 1;
    std::int64_t output_total_size = 1;
    for (auto inputport : node->InputPorts()) {
        std::int64_t input_size = 1;
        framework::shape_t shape = inputport.entity.tensor.shape;
        input_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        input_size *= 4;
        input_total_size += input_size;
    }
    node->InputMemory(input_total_size);
    // 输出数据大小
    for (auto outputport : node->OutputPorts()) {
        std::int64_t output_size = 1;
        framework::shape_t shape = outputport.entity.shape;
        output_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
        output_size *= 4;
        output_total_size += output_size;
    }
    node->OutputMemory(output_total_size);
    // 节点结构大小
    node->PersistentMemory(sizeof(*node));
}

void CalculateCost(std::shared_ptr<framework::NodeBase> node) {
    std::int64_t compute_cost = ceil(
        0.2
        * ceil((node->InputMemory() + node->OutputMemory())
               / (1 + exp(ceil(-(fabs(node->InputMemory() - node->OutputMemory())) / (1 + node->OutputMemory()))))));
    node->ComputeCost(compute_cost);
}

float GetFactor() {
    const char* fetch_node = getenv("TARGET_FACTOR");
    if (!fetch_node) {
        SPDLOG_WARN("batch size is not set");
        return 1.2;
    }
    return std::stof(fetch_node);
}

std::string GetGraphJson() {
    const char* fetch_node = getenv("GRAPH");
    if (!fetch_node) {
        SPDLOG_WARN("graph path is not set");
        return "";
    }
    return std::string{fetch_node};
}

int GetOOM() {
    const char* fetch_node = getenv("OOM");
    if (!fetch_node) {
        SPDLOG_WARN("OOM is not set");
        return 1;
    }
    return std::stoi(fetch_node);
}

int GetDeviceNum() {
    const char* fetch_node = getenv("DEVICE_NUM");
    if (!fetch_node) {
        SPDLOG_WARN("device num is not set");
        return 2;
    }
    return std::stoi(fetch_node);
}

std::string GetPlacementRpcAddressEnvVar() {
    const char* address = getenv("TF_PLACEMENT_RPC_ADDRESS");
    if (!address) {
        SPDLOG_WARN("TF_PLACEMENT_RPC_ADDRESS is not set");
        return "";
    }
    return std::string{address};
}

enum class PolicyType {
    None,
    Aware,
    FdDps,
    SGP,
    Trinity,
};

PolicyType GetPlacementPolicyVar() {
    const char* policy = getenv("TF_PLACEMENT_POLICY");
    if (!policy) {
        return PolicyType::None;
    }
    std::string policy_str{policy};
    if (policy_str == "aware") {
        return PolicyType::Aware;
    }
    if (policy_str == "fddps") {
        return PolicyType::FdDps;
    }
    if (policy_str == "SGP") {
        return PolicyType::SGP;
    }
    if (policy_str == "trinity") {
        return PolicyType::Trinity;
    }
    return PolicyType::None;
}

struct ConvertContext {
    std::map<std::string, const NodeDef*> name_to_node;
    std::map<std::string, std::vector<std::pair<const NodeDef*, int>>> name_to_output;

    explicit ConvertContext(const GraphDef& graphdef) {
        for (const auto& node_def : graphdef.node()) {
            name_to_node.insert({node_def.name(), &node_def});
        }
        for (const auto& node_def : graphdef.node()) {
            for (const auto& it : node_def.input()) {
                auto view = it | ranges::views::split(':') | ranges::to<std::vector<std::string>>();
                auto output = name_to_output.find(view[0]);
                if (output != name_to_output.end()) {
                    auto& outputs = output->second;
                    outputs.emplace_back(name_to_node[node_def.name()], view.size() == 1 ? 0 : std::stoi(view[1]));
                } else {
                    name_to_output.insert(
                        {view[0], {{name_to_node[node_def.name()], view.size() == 1 ? 0 : std::stoi(view[1])}}});
                }
            }
        }
    }
};

void SetDevice(Graph& g, std::map<std::string, std::string> node_to_device) {
    for (int i = 0; i < g.num_node_ids(); ++i) {
        auto* node = g.FindNodeId(i);
        auto find_iter = node_to_device.find(node->name());
        if (find_iter != node_to_device.end()) {
            node->set_requested_device(find_iter->second);
        }
    }
}

// cppcheck-suppress constParameterReference
std::map<std::string, std::string> GetDeviceMapFromCostNodes(std::vector<framework::CostNode>& nodes) {
    return nodes | ranges::views::transform([](auto& a) { return std::make_pair(a.GetName(), a.GetDevice()); })
           | ranges::to<std::map<std::string, std::string>>();
}

framework::Graph ConvertGraphDefToGraph(const GraphDef& graph_def) {
    framework::Graph graph;
    auto context = ConvertContext(graph_def);
    Json::Reader reader;
    Json::Value root;
    std::ifstream is;
    is.open(GetGraphJson(), std::ios::binary);
    reader.parse(is, root);
    for (const auto& node_def : graph_def.node()) {
        auto jsonnode = root[node_def.name()];
        framework::NodeBase node;
        node.Attrs().insert(
            std::pair<std::string, std::string>("colocation_group", jsonnode["colocation_group"].asString()));
        node.Name(node_def.name());
        node.Op(node_def.op());
        node.Device(node_def.device());
        for (int id = 0; id < node_def.input_size(); id++) {
            const std::string& input = node_def.input(id);
            auto view = input | ranges::views::split(':') | ranges::to<std::vector<std::string>>();
            std::string index = "0";
            if (view.size() == 2) {
                index = view[1];
            }
            std::string input_node = view[0];
            framework::shape_t shape;
            framework::DataType dtype;
            int indexi = std::stoi(index);
            if (input_node.rfind('^', 0) == 0) {
                input_node = input_node.substr(1);
            }
            const auto& inputshape = jsonnode["inputs"][input_node]["shape"];
            std::transform(inputshape.begin(), inputshape.end(), std::back_inserter(shape),
                           [](const auto& dim) { return dim.asInt(); });
            dtype = type_map[jsonnode["inputs"][input_node]["dtype"].asString()];
            auto result = node.AddInputPort(input_node, indexi, id, dtype, shape);
            node.AddInput(input);
        }
        auto sorted_outputs = std::move(context.name_to_output[node_def.name()]) | ranges::actions::unique
                              | ranges::actions::sort([](const auto& a, const auto& b) { return a.second < b.second; });
        for (const auto& i : sorted_outputs) {
            node.AddOutput(i.first->name());
            framework::shape_t shape;
            framework::DataType dtype;
            const auto& outputshape = jsonnode["outputs"][std::to_string(i.second)]["shape"];
            std::transform(outputshape.begin(), outputshape.end(), std::back_inserter(shape),
                           [](const auto& dim) { return dim.asInt(); });
            dtype = type_map[jsonnode["outputs"][std::to_string(i.second)]["dtype"].asString()];
            auto result = node.AddOutputPort(dtype, shape, i.second);
        }
        node.OutputsNum(node.Outputs().size());
        graph.AddNode(node);
    }
    for (auto node : graph.Nodes()) {
        CalculateMemory(node);
        CalculateCost(node);
    }
    return graph;
}

// cppcheck-suppress unusedFunction
Status PlacementPass::Run(const GraphOptimizationPassOptions& options) {
    SPDLOG_INFO("PlacementPass is_function_graph {}", options.is_function_graph);

    GraphDef graph_def;
    (*options.graph)->ToGraphDef(&graph_def);
    auto graph = ConvertGraphDefToGraph(graph_def);

    std::map<std::string, std::string> device_map;
    auto policy = GetPlacementPolicyVar();
    if (policy == PolicyType::None) {
        SPDLOG_WARN("F_PLACEMENT_POLICY is not set. skip PlacementPass.");
        return Status::OK();
    }

    if (policy == PolicyType::Aware || policy == PolicyType::Trinity) {
        auto rpc_address = GetPlacementRpcAddressEnvVar();
        for (const auto& node : graph.Nodes()) {
            std::int64_t comm_cost_input = 1;
            std::int64_t comm_cost_output = 1;
            std::map<std::string, std::string> attr = node->Attrs();
            for (std::string input : node->Inputs()) {
                if (attr.count("input:" + input) > 0) {
                    std::string shape = attr.at("input:" + input);
                    if (!shape.empty()) {
                        std::string space = " ";
                        std::vector<std::int64_t> number;
                        size_t pos = 0;
                        while ((pos = shape.find(space)) != string::npos) {
                            number.push_back(atoi(shape.substr(0, pos).c_str()));
                            shape.erase(0, pos + space.length());
                        }
                        std::int64_t comm_cost =
                            std::accumulate(number.begin(), number.end(), 1, std::multiplies<int>());
                        comm_cost *= sizeof(attr["type"]);
                        comm_cost_input += comm_cost;
                    }
                }
            }
            node->PersistentMemory(comm_cost_input);
            for (std::string output : node->Outputs()) {
                if (attr.count("output:" + output) > 0) {
                    std::string shape = attr.at("output:" + output);
                    if (!shape.empty()) {
                        std::string space = " ";
                        std::vector<std::int64_t> number;
                        size_t pos = 0;
                        while ((pos = shape.find(space)) != string::npos) {
                            number.push_back(atoi(shape.substr(0, pos).c_str()));
                            shape.erase(0, pos + space.length());
                        }
                        std::int64_t comm_cost =
                            std::accumulate(number.begin(), number.end(), 1, std::multiplies<int>());
                        comm_cost *= sizeof(attr["type"]);
                        comm_cost_output += comm_cost;
                    }
                }
            }
            node->OutputMemory(comm_cost_output);
            std::int64_t compute_cost =
                ceil(0.2
                     * ceil((node->PersistentMemory() + node->OutputMemory())
                            / (1
                               + exp(ceil(-(fabs(node->PersistentMemory() - node->OutputMemory()))
                                          / (1 + node->OutputMemory()))))));
            node->ComputeCost(compute_cost);
        }
        if (rpc_address.empty()) {
            SPDLOG_WARN("TF_PLACEMENT_RPC_ADDRESS is not set. skip!");
            return Status::OK();
        }
        auto rpc_graph = framework::ConvertGraphToMessage(graph);

        framework::RpcServiceClient client(grpc::CreateChannel(rpc_address, grpc::InsecureChannelCredentials()));
        const char* policy_ = getenv("TF_PLACEMENT_POLICY");
        std::string policy_str{policy_};
        auto r = client.Call(rpc_graph, policy_str);
        if (r.has_error()) {
            SPDLOG_WARN("call rpc error. {}", r.error().text);
            return Status::OK();
        }
        device_map = std::move(r.value());
        SPDLOG_INFO(fmt::to_string(device_map));
    } else if (policy == PolicyType::FdDps) {
        std::vector<framework::Device> devices;
        for (auto* i : options.device_set->devices()) {
            auto memory = i->attributes().memory_limit();
            devices.emplace_back(framework::DeviceTypeFrom(i->device_type()), i->name(), memory, memory, 0);
        }
        framework::CostGraph cost_graph = ConvertGraphToCostGraph(graph);
        framework::FDDPSAlgorithm fddps_algorithm(cost_graph, devices);
        auto r = fddps_algorithm.Placement();
        if (r.has_error()) {
            SPDLOG_INFO("call fddps error. {}", r.error().text);
            return Status::OK();
        }
        device_map = GetDeviceMapFromCostNodes(r.value());
    } else if (policy == PolicyType::SGP) {
        std::vector<framework::Device> devices;
        for (auto* i : options.device_set->devices()) {
            auto memory = i->attributes().memory_limit();
            if (i->device_type() != "CPU") {
                devices.emplace_back(framework::DeviceTypeFrom(i->device_type()), i->name(), memory, memory, 0);
            }
        }
        framework::Partition Partition(graph, GetDeviceNum(), devices, GetFactor(), GetOOM());
        device_map = Partition.op_group;
    }

    SetDevice(**options.graph, device_map);
    return Status::OK();
}

REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1, PlacementPass);
}  // namespace tensorflow
