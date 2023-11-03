#pragma once
#include <iostream>
#include <memory>
#include <string>

#include "adapters/tensorflow/rpc/graph.pb.h"
#include "adapters/tensorflow/rpc/service.grpc.pb.h"
#include "adapters/tensorflow/rpc/util.h"
#include "common/error.hpp"
#include "fmt/format.h"
#include "grpcpp/grpcpp.h"
#include "result.hpp"
#ifndef ADAPTERS_TENSORFLOW_RPC_CLIENT_H
#define ADAPTERS_TENSORFLOW_RPC_CLIENT_H
using framework::rpc::CallRequest;
using framework::rpc::CallResponse;
using framework::rpc::RpcService;
using RpcGraph = framework::rpc::Graph;
using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
namespace framework {

/**
 * RpcServiceClient
 *
 * client api
 */
class RpcServiceClient {
  public:
    explicit RpcServiceClient(std::shared_ptr<Channel> channel);

    /**
     * call strategy rpc
     */
    cpp::result<std::map<std::string, std::string>, Error> Call(const RpcGraph& graph, std::string policy);

  private:
    std::unique_ptr<RpcService::Stub> stub_;
};
}  // namespace framework
#endif
