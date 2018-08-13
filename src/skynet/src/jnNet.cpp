
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

"BeginNet":                             ///< вход сети
{
"NextNodes": ....,                      ///< имена след узлов через пробел
},

"AuxInputs":                            ///< вспом входы сети. Необязательно
[
{
"NodeName":....,                        ///< имя узла. Дбыть уникальным в пределах ветви, без ' ' и '-'.
"NextNodes": ....,                      ///< имена след узлов через пробел
},
]

"Nodes":                                ///< узлы ветви
[
{
"NodeName":....,                        ///< имя узла. Дбыть уникальным в пределах ветви, без ' ' и '-'.
"NextNodes": ....,                      ///< имена след узлов через пробел. Для условного перехода: оператор узла должен выдать имя след узла, иначе пойдет на все
"OperatorName":....,                    ///< имя оператора узла. Дбыть определен в FNOperator.dll.
"OperatorParams": {"name":"value",}     ///< Необязательно. Список параметров оператора, параметры индивидуальны для каждого оператора. Если опустить, будут по умолч.
},
],

"AuxOutputs":                           ///< вспом выхода сети
[
{
"NodeName":....,                        ///< имя узла. Дбыть уникальным в пределах ветви, без ' ' и '-'.
"PrevNode": ....,                       ///< предыд узел. Мбыть только один!
},
]

"EndNet":                               ///< выход сети
{
"PrevNode": ....,                       ///< предыд узел. Мбыть только один!
}
}
*/

/// проверка документа на соотв-е
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
	else{
		err = "!jnDoc.HasMember('BeginNet')"; return false;
	}

	///////////////


	// проверка узлов

	if (!jnDoc.HasMember("Nodes") || !jnDoc["Nodes"].IsArray()){
		err = "!jnDoc.HasMember('Nodes') || !jnDoc['Nodes'].IsArray()"; return false;
	}


	auto Nodes = jnDoc["Nodes"].GetArray();

	int sz = Nodes.Size();
	if (sz == 0){
		err = "jnDoc['Nodes'].Size() == 0"; return false;
	}

	for (int i = 0; i < sz; ++i){

		auto Node = Nodes[i].GetObject();

		if (!Node.HasMember("NodeName") || !Node["NodeName"].IsString()){
			err = "!Node.HasMember('NodeName') || !Node['NodeName'].IsString()"; return false;
		}

		if (!Node.HasMember("NextNodes") || !Node["NextNodes"].IsString()){
			err = "!Node.HasMember('NextNodes') || !Node['NextNodes'].IsString()"; return false;
		}

		if (!Node.HasMember("OperatorName") || !Node["OperatorName"].IsString()){
			err = "!Node.HasMember('OperatorName') || !Node['OperatorName'].IsString()"; return false;
		}
	}

	///////////////
		
	// проверка Output
	if (jnDoc.HasMember("EndNet")){

		if (!jnDoc["EndNet"].IsObject()){
			err = "!jnDoc['EndNet'].IsObject()"; return false;
		}

		auto EndNet = jnDoc["EndNet"].GetObject();

		if (!EndNet.HasMember("PrevNode") || !EndNet["PrevNode"].IsString()){
			err = "!EndNet.HasMember('PrevNode') || !EndNet['PrevNode'].IsString()"; return false;
		}
	}
	else{
		err = "!jnDoc.HasMember('EndNet')"; return false;
	}
	///////////////
		
	return true;
}

/// узлы ветви
bool jnGetNodes(rapidjson::Document& jnDoc, std::map<std::string, Node>& out_nodes, string& out_err){

	// Создаем узлы
	auto nodes = jnDoc["Nodes"].GetArray();
	int sz = nodes.Size();
	for (int i = 0; i < sz; ++i){

		auto node = nodes[i].GetObject();

		Node nd;
		nd.name = trim(node["NodeName"].GetString());
		nd.oprName = trim(node["OperatorName"].GetString());
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

	// проверим, что ссылка на первый узел только одна
	int beginRefCnt = 0;
	for (auto& n : out_nodes){

		if (find(n.second.prevNodes.begin(), n.second.prevNodes.end(), "BeginNet") != n.second.prevNodes.end())
			++beginRefCnt;
	}
	if (beginRefCnt > 1){
		out_err = string("'BeginNet' ref must be only one");
		return false;
	}
	
	// проверим, что ссылка на последний узел только одна
	int endRefCnt = 0;
	for (auto& n : out_nodes){

		if (find(n.second.nextNodes.begin(), n.second.nextNodes.end(), "EndNet") != n.second.nextNodes.end())
			++endRefCnt;
	}
	if (endRefCnt > 1){
		out_err = string("'EndNet' ref must be only one");
		return false;
	}

	// заполняем prevNodes
	for (auto& on : out_nodes){

		for (auto& nn : on.second.nextNodes){

			out_nodes[nn].prevNodes.push_back(on.first);
		}
	}

	return true;
}

/// начало ветви
bool jnGetBegin(rapidjson::Document& jnDoc, std::map<std::string, Node>& out_nodes, string& out_err){

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

	return true;
}

/// конец ветви
bool jnGetEnd(rapidjson::Document& jnDoc, std::map<std::string, Node>& out_nodes, string& out_err){

	Node nd;
	nd.name = "EndNet";
	nd.oprName = "Output";
	
	auto EndNet = jnDoc["EndNet"].GetObject();		
	if (split(trim(EndNet["PrevNode"].GetString()), " ").size() > 1){
		out_err = "outputBr.EndNet['PrevNode'].size() > 1";
		return false;
	}	

	out_nodes["EndNet"] = nd;

	return true;
}
	
// парсинг структуры сети
bool SNet::jnParseNet(const std::string& branchJSON, SN_Base::Net& out_net, std::string& out_err){

	jnNet_.Parse(branchJSON.c_str());

	// Проверка 
	if (!jnCheckJDoc(jnNet_, out_err)) return false;

	// начало сети
	if (!jnGetBegin(jnNet_, out_net.nodes, out_err)) return false;

	// конец сети
	if (!jnGetEnd(jnNet_, out_net.nodes, out_err)) return false;

	// получаем узлы
	if (!jnGetNodes(jnNet_, out_net.nodes, out_err)) return false;
					
	return true;
}

/// задать параметры узла
bool SNet::setParamNode(const char* nodeName, const char* jnParam){
	std::unique_lock<std::mutex> lk(mtxCmn_);

	//rj::Document jnDoc;
	//jnDoc.Parse(jnParam);

	//if (!jnDoc.IsObject()){
	//	statusMess("setParamNode error: !jnDoc.IsObject()");
	//	return false;
	//}

	//if (!jnDoc.HasMember("OperatorParams") || !jnDoc["OperatorParams"].IsObject()){
	//	statusMess("setParamNode error: !jnDoc.HasMember("OperatorParams") || !jnDoc["OperatorParams"].IsObject()");
	//	return false;
	//}

	//auto oprPrms = node["OperatorParams"].GetObject();

	//const char* kTypeNames[] = { "Null", "False", "True", "Object", "Array", "String", "Number" };

	//for (rj::Value::ConstMemberIterator itr = oprPrms.MemberBegin(); itr != oprPrms.MemberEnd(); ++itr){

	//	if (string(kTypeNames[itr->value.GetType()]) == "String")
	//		nd.oprPrms[trim(itr->name.GetString())] = trim(itr->value.GetString());
	//	else{
	//		out_err = string("node['OperatorParams'][") + itr->name.GetString() + "] != 'String'";
	//		return false;
	//	}
	//}

	return true;
}

/// вернуть параметры узла
bool SNet::getParamNode(const char* nodeName, char* jnParam /*minsz 256*/){
	std::unique_lock<std::mutex> lk(mtxCmn_);

	auto nodes = jnNet_["Nodes"].GetArray();
	int sz = nodes.Size();
	for (int i = 0; i < sz; ++i){

		auto node = nodes[i].GetObject();

		if ((string(node["NodeName"].GetString()) == nodeName) && node.HasMember("OperatorParams")){

			rapidjson::StringBuffer sb;
			rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
			node["OperatorParams"].Accept(writer);
					
			strcpy(jnParam, sb.GetString());
			return true;
		}
	}

	return false;
}

/// вернуть архитектуру сети
bool SNet::getArchitecNet(char* jnArchitecNet /*minsz 2048*/){
	std::unique_lock<std::mutex> lk(mtxCmn_);

	strcpy(jnArchitecNet, jnNet_.GetString());

	return true;
}