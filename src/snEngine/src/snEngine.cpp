//
// sunnet project
// Copyright (C) 2018 by Contributors <https://github.com/Tyill/sunnet>
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
#include <set>
#include "snBase/snBase.h"
#include "snEngine/snEngine.h"

using namespace std;
using namespace SN_Base;

//#ifdef SN_DEBUG
//#define SN_ENG_DMESS(mess) statusMess(mess);
//#define SN_ENG_PRINTMESS(mess) printf("%s \n", mess);
//#else
#define SN_ENG_DMESS(mess);
#define SN_ENG_PRINTMESS(mess);
//#endif

namespace SN_Eng{

    void SNEngine::statusMess(const std::string& mess){

        if (stsCBack_) stsCBack_(mess);
    }
        
    /// создание потоков для forward
    void SNEngine::createThreadsFwd(std::map<std::string, SN_Base::Node>& nodes, ThreadPool* thrPool){
               
        set<string> nodeExist;

        for (auto& n : nodes){
            if (n.second.oprName == "Input"){

                thrPool->addNode(n.first);
                thrPool->addThread(n.first);
                nodeExist.insert(n.first);

                ndStates_[n.first].parentFW = n.first;
            }
        }

        for (auto& n : nodes){

            if (nodeExist.find(n.first) != nodeExist.end()) continue;

            if (n.second.nextNodes.size() > 1){
                for (auto& nn : n.second.nextNodes){

                    if (nodeExist.find(nn) != nodeExist.end()) continue;

                    thrPool->addNode(nn);
                    nodeExist.insert(nn);
                }
            }            
            if (n.second.prevNodes.size() > 1){                
                thrPool->addNode(n.first);
                nodeExist.insert(n.first);
            }
            else{
                if (nodes[n.second.prevNodes[0]].nextNodes.size() > 1){
                    thrPool->addNode(n.first);
                    nodeExist.insert(n.first);
                }
            }
        }

        for (auto& n : ndStates_){

            if (n.second.parentFW.empty()){
               
                /// найдем начало цепочки - только там замок
                string firts = n.first;
                while (nodes[firts].prevNodes.size() == 1){

                    string& prev = nodes[firts].prevNodes[0];
                    auto& nn = nodes[prev].nextNodes;
                    if (nn.size() > 1) break;            ///< развилка?

                    if (!ndStates_[prev].parentFW.empty()){
                        firts = ndStates_[prev].parentFW;
                        break;
                    }

                    firts = prev;   ///< идем обратно дальше
                }

                n.second.parentFW = firts;
            }
        }

        /// подождем пока все запустятся
        for (auto& n : nodes){
            if (n.second.oprName == "Input"){
                thrPool->waitExist(n.first);
            }
        }                      
    }

    /// создание потоков для backward
    void SNEngine::createThreadsBwd(std::map<std::string, SN_Base::Node>& nodes, ThreadPool* thrPool){

        set<string> nodeExist;
               
        operParam_.action = SN_Base::snAction::backward;

        for (auto& n : nodes){
            if (n.second.oprName == "Output"){
                
                thrPool->addNode(n.first);
                thrPool->addThread(n.first);
                nodeExist.insert(n.first);

                ndStates_[n.first].parentBW = n.first;
            }
        }

        for (auto& n : nodes){

            if (nodeExist.find(n.first) != nodeExist.end()) continue;

            if (n.second.prevNodes.size() > 1){
                for (auto& nn : n.second.prevNodes){

                    if (nodeExist.find(nn) != nodeExist.end()) continue;

                    thrPool->addNode(nn);
                    nodeExist.insert(nn);
                }
            }
            if (n.second.nextNodes.size() > 1){
                thrPool->addNode(n.first);
                nodeExist.insert(n.first);
            }
            else{
                if (nodes[n.second.nextNodes[0]].prevNodes.size() > 1){
                    thrPool->addNode(n.first);
                    nodeExist.insert(n.first);
                }
            }
        }

        for (auto& n : ndStates_){

            if (n.second.parentBW.empty()){

                /// найдем начало цепочки - только там замок
                string firts = n.first;
                while (nodes[firts].nextNodes.size() == 1){

                    string& nxt = nodes[firts].nextNodes[0];
                    auto& pn = nodes[nxt].prevNodes;
                    if (pn.size() > 1) break;        ///< развилка?

                    if (!ndStates_[nxt].parentBW.empty()){
                        firts = ndStates_[nxt].parentBW;
                        break;
                    }

                    firts = nxt;   ///< идем обратно дальше
                }

                n.second.parentBW = firts;
            }
        }

        /// подождем пока все запустятся
        for (auto& n : nodes){
            if (n.second.oprName == "Output"){

                thrPool->waitExist(n.first);
            }
        }
    }
    
    SNEngine::SNEngine(Net& brNet, 
        std::function<void(const std::string&)> sts) : stsCBack_(sts){

        operats_ = brNet.operats;
        nodes_ = brNet.nodes;
        for (auto& n : brNet.nodes)
            ndStates_[n.first] = ndState();
    }
    
    SNEngine::~SNEngine(){

        fWorkEnd_ = true;
        if (thrPoolForward_) delete thrPoolForward_;
        if (thrPoolBackward_) delete thrPoolBackward_;
    }
            
    /// прямой проход
    bool SNEngine::forward(const SN_Base::operationParam& operPrm){
        
        operParam_ = operPrm;

        if (!thrPoolForward_){
            thrPoolForward_ = new ThreadPool(bind(&SNEngine::operatorThreadForward, this, std::placeholders::_1));
            createThreadsFwd(nodes_, thrPoolForward_);
        }

        /// предварительно установим готовность
        thrPoolForward_->preStartAll();
        for (auto& n : nodes_)
            ndStates_[n.first].isWasRun = false;

        /// запускаем прямой ход 
        for (auto& n : nodes_){
            if (n.second.oprName == "Input")
                thrPoolForward_->startTask(n.first);
        }
        
        /// ждем, когда протечет до конца
        thrPoolForward_->waitAll();
                
        return true;
    }

    /// обратный проход
    bool SNEngine::backward(const SN_Base::operationParam& operPrm){
                                
        operParam_ = operPrm;

        if (!thrPoolBackward_){
            thrPoolBackward_ = new ThreadPool(bind(&SNEngine::operatorThreadBackward, this, std::placeholders::_1));
            createThreadsBwd(nodes_, thrPoolBackward_);
        }

        /// предварительно установим готовность
        thrPoolBackward_->preStartAll();
        for (auto& n : nodes_)
            if (!ndStates_[n.first].isWasRun) thrPoolBackward_->resetPrestart(n.first);

        /// запускаем обр ход 
        for (auto& n : nodes_){
            if (n.second.oprName == "Output")
                thrPoolBackward_->startTask(n.first);
        }        
            
        /// ждем, когда протечет до конца
        thrPoolBackward_->waitAll();
        
        return true;
    }
    
    /// выполнение оператора при движении вперед
    void SNEngine::actionForward(map<string, Node>& nodes, const string& nname){
        if (fWorkEnd_) return;
                
        /// предыдущие узлы
        auto& prevNodes = nodes[nname].prevNodes;

        if (prevNodes.size() == 1){
            
            /// выполнение оператора
            auto& pn = prevNodes[0];                   SN_ENG_DMESS("node " + nname + " single prevNode " + pn)
            ndStates_[nname].selectNextNodes = operats_[nname]->Do(operParam_, vector<OperatorBase*>{operats_[pn]});
        }
        else{
            /// собираем все предыд операторы
                        
            /// подождем, что все предыд операторы выполнились
            for (auto& n : prevNodes){
            
                string& firts = ndStates_[n].parentFW;               
                                                       SN_ENG_DMESS("node " + nname + " waitFinish thread " + firts)
                thrPoolForward_->waitFinish(firts);    
            }

            /// проверим, что предыд оператор выбирал этот узел
            vector<OperatorBase*> neighb;
            for (auto& n : prevNodes){
                
                if (!ndStates_[n].isWasRun) continue;
                
                auto& prevSelNodes = ndStates_[n].selectNextNodes;

                bool isSel = prevSelNodes.empty() ||
                    (find(prevSelNodes.begin(), prevSelNodes.end(), nname) != prevSelNodes.end());
                                
                if (isSel){                   
                    neighb.push_back(operats_[n]);    SN_ENG_DMESS("node " + nname + " multi prevNode " + n)
                }
            }
            
            /// выполнение оператора
            SN_ENG_DMESS("node " + nname + " actionForward multi")
            ndStates_[nname].selectNextNodes = operats_[nname]->Do(operParam_, neighb);
        }
        ndStates_[nname].isWasRun = true;
    }

    /// выполнение оператора при движении назад
    void SNEngine::actionBackward(std::map<std::string, Node>& nodes, const std::string& nname){
        if (fWorkEnd_) return;

        /// если только один путь
        if (nodes[nname].nextNodes.size() == 1){            
            
            /// выполнение оператора
            auto& nd = nodes[nname].nextNodes[0];         SN_ENG_DMESS("node " + nname + " single prevNode " + nd)
            operats_[nname]->Do(operParam_, vector<OperatorBase*>{operats_[nd]});
        }
        else{
            /// следующие узлы, которые были выбраны на пред итерации
            auto& prevNextNodes = ndStates_[nname].selectNextNodes;
            
            /// подождем, что все пред операторы выполнились
            for (auto& n : prevNextNodes){
                
                string& firts = ndStates_[n].parentBW;

                if (!ndStates_[firts].isWasRun) continue;
                                                         SN_ENG_DMESS("node " + nname + " waitFinish thread " + firts)
                thrPoolBackward_->waitFinish(firts);     
            }
            

            /// собираем все пред операторы
            vector<OperatorBase*> neighb;
            for (auto& n : prevNextNodes){
                neighb.push_back(operats_[n]);           SN_ENG_DMESS("node " + nname + " multi prevNode " + n)
            }
            
            /// выполнение оператора           
            operats_[nname]->Do(operParam_, neighb);     SN_ENG_DMESS("node " + nname + " multi")
        }
    }
            
    /// выбор след узла при движении вперед
    std::string SNEngine::selectNextForward(std::map<std::string, Node>& nodes, const std::string& nname, std::string& nnameMem){
        if (fWorkEnd_) return nnameMem;

        auto& nextNodes = nodes[nname].nextNodes;

        string selWay = "";

        /// путь только один?
        if (nextNodes.size() == 1){

            string& nn = nextNodes[0];

            /// дальше идти?
            if (!ndStates_[nname].selectNextNodes.empty() && 
                (ndStates_[nname].selectNextNodes[0] == "noWay")){
                               
                /// тек поток тормозим
                thrPoolForward_->finish(nnameMem);        SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
            }            
            /// у след узла один предыдущий?
            else if (nodes[nn].prevNodes.size() == 1){

                /// продолжение пути оставляем в этом же потоке
                selWay = nn;                              SN_ENG_DMESS("node " + nname + " selWay " + nn)
            }
            else{
                /// рестарт в том же потоке
                thrPoolForward_->restartTask(nnameMem, nn); SN_ENG_DMESS("node " + nname + " restart thread " + nn)
            }
        }
        /// конец пути?
        else if (nextNodes.empty()){
                       
            /// тек поток тормозим
            thrPoolForward_->finish(nnameMem);            SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
        }
        else{
            /// выбираем дальнейший путь в завис от результа, который вернул текущий оператор
            vector<string> nextWays, &selectNextNodes = ndStates_[nname].selectNextNodes;
            if (selectNextNodes.empty()){
                nextWays = nextNodes;
                ndStates_[nname].selectNextNodes = nextNodes;   ///< значит, выбрали все
            }
            else{
                /// выбираем известные пути
                for (auto& way : nextNodes){
                    if (find(selectNextNodes.begin(), selectNextNodes.end(), way) != selectNextNodes.end())
                        nextWays.push_back(way);
                }
            }
                                            
            /// ответвления расталкиваем по другим потокам
            bool isSelNextWay = false;
            for (auto& nw : nextWays){

                /// продолжение пути оставляем в этом же потоке
                if (!isSelNextWay){

                    /// рестарт в том же потоке
                    thrPoolForward_->restartTask(nnameMem, nw); SN_ENG_DMESS("node " + nname + " restart thread " + nw)

                    isSelNextWay = true;
                    continue;
                }

                thrPoolForward_->startTask(nw);                 SN_ENG_DMESS("node " + nname + " start thread " + nw)
            }                                                   
                                                                
            if (nextWays.empty()){                              
                /// тек поток тормозим                          
                thrPoolForward_->finish(nnameMem);              SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
            }
        }

        return selWay;
    }

    /// выбор след узла при движении назад
    std::string SNEngine::selectNextBackward(std::map<std::string, Node>& nodes, const std::string& nname, std::string& nnameMem){
        if (fWorkEnd_) return nnameMem;

        auto& prevNodes = nodes[nname].prevNodes;

        string selWay = "";

        /// путь только один?
        if (prevNodes.size() == 1){

            string& pn = prevNodes[0];

            /// дальше идти?
            if (!ndStates_[pn].selectNextNodes.empty() && 
                (ndStates_[pn].selectNextNodes[0] == "noWay")){
                
                /// тек поток тормозим
                thrPoolBackward_->finish(nnameMem);          SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
            }
            /// у пред узла один следующий?
            else if(nodes[pn].nextNodes.size() == 1){

                /// продолжение пути оставляем в этом же потоке
                selWay = pn;                                 SN_ENG_DMESS("node " + nname + " selWay " + pn)
            }
            else{                
                /// рестарт в том же потоке
                thrPoolBackward_->restartTask(nnameMem, pn); SN_ENG_DMESS("node " + nname + " restart thread " + pn)
            }
        }
        /// конец пути?
        else if (prevNodes.empty()){
       
            /// тек поток тормозим
            thrPoolBackward_->finish(nnameMem);              SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
        }
        else{
            /// выбираем дальнейший путь в завис от результа, который вернул предыд оператор
            vector<string> nextWays;
            for (auto& way : prevNodes){

                if (!ndStates_[way].isWasRun) continue;

                auto& selNextNodes = ndStates_[way].selectNextNodes;
                if (selNextNodes.empty() || (find(selNextNodes.begin(), selNextNodes.end(), nname) != selNextNodes.end()))
                    nextWays.push_back(way);
            }
                    
            /// ответвления расталкиваем по другим потокам
            bool isSelNextWay = false;
            for (auto& nw : nextWays){

                /// продолжение пути оставляем в этом же потоке
                if (!isSelNextWay){
                   
                    /// рестарт в том же потоке
                    thrPoolBackward_->restartTask(nnameMem, nw); SN_ENG_DMESS("node " + nname + " restart thread " + nw)

                    isSelNextWay = true;
                    continue;
                }

                thrPoolBackward_->startTask(nw);                 SN_ENG_DMESS("node " + nname + " start thread " + nw)
            }

            if (nextWays.empty()){
                /// тек поток тормозим
                thrPoolBackward_->finish(nnameMem);              SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
            }
        }

        return selWay;
    }

    /// рабочий поток для каждой нити операторов
    void SNEngine::operatorThreadForward(std::string nname){
        
        std::string nnameMem = nname;
        
        nname.clear();
                                    
        while (!fWorkEnd_){
               
            /// ждем след итерацию
            if (nname.empty())
              nname = thrPoolForward_->waitStart(nnameMem);

            /// обработка текущего узла
            actionForward(nodes_, nname);

            /// выбор следующего узла 
            nname = selectNextForward(nodes_, nname, nnameMem);                
        }        
    }

    /// рабочий поток для каждой нити операторов
    void SNEngine::operatorThreadBackward(std::string nname){

        std::string nnameMem = nname;

        nname.clear();
                
        while (!fWorkEnd_){

            /// ждем след итерацию
            if (nname.empty())
              nname = thrPoolBackward_->waitStart(nnameMem);

            /// обработка текущего узла
            actionBackward(nodes_, nname);

            /// выбор следующего узла 
            nname = selectNextBackward(nodes_, nname, nnameMem);
        }
    }
}
