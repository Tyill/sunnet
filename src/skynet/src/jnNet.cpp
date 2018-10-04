//
// SkyNet Project
// Copyright (C) 2018 by Contributors <https://github.com/Tyill/skynet>
//
// This code is licensed under the MIT License.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include <algorithm>
#include "snBase/snBase.h"
#include "snAux/auxFunc.h"
#include "Lib/rapidjson/document.h"
#include "Lib/rapidjson/stringbuffer.h"
#include "Lib/rapidjson/writer.h"
#include "snet.h"

using namespace std;
using namespace SN_Aux;
using namespace SN_Base;

namespace rj = rapidjson;


/*
{

"BeginNet":                             ///< input net
{
"NextNodes": ....,                      ///< names next node's
},

"Nodes":                                ///< node's of net
[
{
"NodeName":....,                        ///< name node. Should be unique, without ' ' и '-'.
"NextNodes": ....,                      ///< names of the trace nodes through a space. For a conditional jump: the node operator must issue the node trace name, otherwise it will go to all
"OperatorName":....,                    ///< name of the host operator. Must be defined in snOperator.lib
"OperatorParams": {"name":"value",}     ///< Not necessary. List of operator parameters, parameters are individual for each operator. If to omit, will be by default.
},
],

"EndNet":                               ///< output of net
{
"PrevNode": ....,                       ///< prev node. Can be only one
}
}
*/

/// document validation
bool jnCheckJDoc(rapidjson::Document& jnDoc, string& err){

    if (!jnDoc.IsObject()){
        err = "!jnDoc.IsObject() errOffset " + to_string(jnDoc.GetParseError()); return false;
    }
    
    if (jnDoc.HasMember("BeginNet")){

        if (!jnDoc["BeginNet"].IsObject()){
            err = "!jnDoc['BeginNet'].IsObject()"; return false;
        }

        auto BeginNet = jnDoc["BeginNet"].GetObject();

        if (!BeginNet.HasMember("NextNodes") || !BeginNet["NextNodes"].IsString()){
            err = "!BeginNet.HasMember('NextNodes') || !BeginNet['NextNodes'].IsString()"; return false;
        }
    }   

    ///////////////


    // check nodes

    if (!jnDoc.HasMember("Nodes") || !jnDoc["Nodes"].IsArray()){
        err = "!jnDoc.HasMember('Nodes') || !jnDoc['Nodes'].IsArray()"; return false;
    }


    auto nodes = jnDoc["Nodes"].GetArray();

    int sz = nodes.Size();
    if (sz == 0){
        err = "jnDoc['Nodes'].Size() == 0"; return false;
    }

    for (int i = 0; i < sz; ++i){

        auto node = nodes[i].GetObject();

        if (!node.HasMember("NodeName") || !node["NodeName"].IsString()){
            err = "!Node.HasMember('NodeName') || !Node['NodeName'].IsString()"; return false;
        }
        
        if (!node.HasMember("OperatorName") || !node["OperatorName"].IsString()){
            err = "!Node.HasMember('OperatorName') || !Node['OperatorName'].IsString()"; return false;
        }

        if (node["OperatorName"].GetString() == "Output") continue;

        if (!node.HasMember("NextNodes") || !node["NextNodes"].IsString()){
            err = "!Node.HasMember('NextNodes') || !Node['NextNodes'].IsString()"; return false;
        }
    }

    ///////////////
        
    // check Output
    if (jnDoc.HasMember("EndNet")){

        if (!jnDoc["EndNet"].IsObject()){
            err = "!jnDoc['EndNet'].IsObject()"; return false;
        }

        auto EndNet = jnDoc["EndNet"].GetObject();

        if (!EndNet.HasMember("PrevNode") || !EndNet["PrevNode"].IsString()){
            err = "!EndNet.HasMember('PrevNode') || !EndNet['PrevNode'].IsString()"; return false;
        }
    }
   
    ///////////////
        
    return true;
}

/// nodes
bool jnGetNodes(rapidjson::Document& jnDoc, std::map<std::string, Node>& out_nodes, string& out_err){

    // create nodes
    auto nodes = jnDoc["Nodes"].GetArray();
    int sz = nodes.Size();
    for (int i = 0; i < sz; ++i){

        auto node = nodes[i].GetObject();

        Node nd;
        nd.name = trim(node["NodeName"].GetString());
        nd.oprName = trim(node["OperatorName"].GetString());
        if (nd.oprName != "Output")
           nd.nextNodes = split(trim(node["NextNodes"].GetString()), " ");

        if (node.HasMember("OperatorParams")){

            if (!node["OperatorParams"].IsObject()){
                out_err = "!node['OperatorParams'].IsObject()";
                return false;
            }

            auto oprPrms = node["OperatorParams"].GetObject();

            const char* kTypeNames[] = { "Null", "False", "True", "Object", "Array", "String", "Number" };

            for (rj::Value::ConstMemberIterator itr = oprPrms.MemberBegin(); itr != oprPrms.MemberEnd(); ++itr){

                if (string(kTypeNames[itr->value.GetType()]) == "String")
                    nd.oprPrms[trim(itr->name.GetString())] = trim(itr->value.GetString());
                else{
                    out_err = string("node['OperatorParams'][") + itr->name.GetString() + "] != 'String'";
                    return false;
                }
            }
        }

        if (out_nodes.find(nd.name) == out_nodes.end())
            out_nodes[nd.name] = nd;
        else{
            out_err = string("NodeName = ") + nd.name + " repeate";
            return false;
        }
    }

    // we check that the link to the first node is only one
    int beginRefCnt = 0;
    for (auto& n : out_nodes){

        if (find(n.second.prevNodes.begin(), n.second.prevNodes.end(), "BeginNet") != n.second.prevNodes.end())
            ++beginRefCnt;
    }
    if (beginRefCnt > 1){
        out_err = string("'BeginNet' ref must be only one");
        return false;
    }
    
    // we check that the link to the last node is only one
    int endRefCnt = 0;
    for (auto& n : out_nodes){

        if (find(n.second.nextNodes.begin(), n.second.nextNodes.end(), "EndNet") != n.second.nextNodes.end())
            ++endRefCnt;
    }
    if (endRefCnt > 1){
        out_err = string("'EndNet' ref must be only one");
        return false;
    }

    // fill prevNodes
    for (auto& on : out_nodes){

        for (auto& nn : on.second.nextNodes){

            out_nodes[nn].prevNodes.push_back(on.first);
        }
    }

    return true;
}

/// begin net
bool jnGetBegin(rapidjson::Document& jnDoc, std::map<std::string, Node>& out_nodes, string& out_err){

    if (jnDoc.HasMember("BeginNet")){

        auto BeginNet = jnDoc["BeginNet"].GetObject();

        Node nd;
        nd.name = "BeginNet";
        nd.oprName = "Input";
        nd.nextNodes = split(trim(BeginNet["NextNodes"].GetString()), " ");

        if (nd.nextNodes.empty()){
            out_err = "input.BeginNet['NextNode'].empty()";
            return false;
        }
                
        out_nodes["BeginNet"] = nd;
    }

    return true;
}

/// end net
bool jnGetEnd(rapidjson::Document& jnDoc, std::map<std::string, Node>& out_nodes, string& out_err){

    if (jnDoc.HasMember("EndNet")){

        Node nd;
        nd.name = "EndNet";
        nd.oprName = "Output";

        auto EndNet = jnDoc["EndNet"].GetObject();

        auto prevNodes = split(trim(EndNet["PrevNode"].GetString()), " ");

        if (prevNodes.empty()){
            out_err = "output.EndNet['PrevNode'].empty()";
            return false;
        }

        if (prevNodes.size() > 1){
            out_err = "output.EndNet['PrevNode'].size() > 1";
            return false;
        }

        out_nodes["EndNet"] = nd;
    }

    return true;
}
    
/// parsing the network structure
bool SNet::jnParseNet(const std::string& branchJSON, SN_Base::Net& out_net, std::string& out_err){

    jnNet_.Parse(branchJSON.c_str());

    if (!jnCheckJDoc(jnNet_, out_err)) return false;

    if (!jnGetBegin(jnNet_, out_net.nodes, out_err)) return false;

    if (!jnGetEnd(jnNet_, out_net.nodes, out_err)) return false;

    if (!jnGetNodes(jnNet_, out_net.nodes, out_err)) return false;
                    
    return true;
}

/// set node parameters
bool SNet::setParamNode(const char* nodeName, const char* jnParam){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    if (operats_.find(nodeName) == operats_.end()){
        statusMess("SN error: '" + string(nodeName) + "' not found");
        return false;
    }
        
    return true;
}

/// get node parameters
bool SNet::getParamNode(const char* nodeName, char** jnParam){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    if (operats_.find(nodeName) == operats_.end()){
        statusMess("SN error: '" + string(nodeName) + "' not found");
        return false;
    }

    auto nodes = jnNet_["Nodes"].GetArray();
    int sz = nodes.Size();
    for (int i = 0; i < sz; ++i){

        auto node = nodes[i].GetObject();

        if ((string(node["NodeName"].GetString()) == nodeName) && node.HasMember("OperatorParams")){

            rapidjson::StringBuffer sb;
            rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
            node["OperatorParams"].Accept(writer);
                    
            *jnParam = (char*)realloc(*jnParam, sb.GetSize() + 1);

            strcpy(*jnParam, sb.GetString());
            return true;
        }
    }

    return false;
}

/// network architecture
bool SNet::getArchitecNet(char** jnArchitecNet){
    std::unique_lock<std::mutex> lk(mtxCmn_);

    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
    jnNet_.Accept(writer);
    
    *jnArchitecNet = (char*)realloc(*jnArchitecNet, sb.GetSize() + 1);
       
    strcpy(*jnArchitecNet, sb.GetString());

    return true;
}
