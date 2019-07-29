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
               
        set<string> thrExist;

        for (auto& n : nodes){
            if (n.second.oprName == "Input"){
                thrPool->addNode(n.first);
                thrExist.insert(n.first);

                ndStates_[n.first].parentFW = n.first;
            }
        }

        for (auto& n : nodes){

            if (thrExist.find(n.first) != thrExist.end()) continue;

            if (n.second.nextNodes.size() > 1){
                for (auto& nn : n.second.nextNodes){

                    if (thrExist.find(nn) != thrExist.end()) continue;

                    thrPool->addNode(nn);
                    thrExist.insert(nn);
                }
            }            
            if (n.second.prevNodes.size() > 1){                
                thrPool->addNode(n.first);
                thrExist.insert(n.first);
            }
            else{
                if (nodes[n.second.prevNodes[0]].nextNodes.size() > 1){
                    thrPool->addNode(n.first);
                    thrExist.insert(n.first);
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
        for (auto& thr : thrExist){
            thrPool->waitExist(thr);
        }      
    }

    /// создание потоков для backward
    void SNEngine::createThreadsBwd(std::map<std::string, SN_Base::Node>& nodes, ThreadPool* thrPool){

        set<string> thrExist;
               
        operParam_.action = SN_Base::snAction::backward;

        for (auto& n : nodes){
            if (n.second.oprName == "Output"){
                thrPool->addNode(n.first);
                thrExist.insert(n.first);

                ndStates_[n.first].parentBW = n.first;
            }
        }

        for (auto& n : nodes){

            if (thrExist.find(n.first) != thrExist.end()) continue;

            if (n.second.prevNodes.size() > 1){
                for (auto& nn : n.second.prevNodes){

                    if (thrExist.find(nn) != thrExist.end()) continue;

                    thrPool->addNode(nn);
                    thrExist.insert(nn);
                }
            }
            if (n.second.nextNodes.size() > 1){
                thrPool->addNode(n.first);
                thrExist.insert(n.first);
            }
            else{
                if (nodes[n.second.nextNodes[0]].prevNodes.size() > 1){
                    thrPool->addNode(n.first);
                    thrExist.insert(n.first);
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
        for (auto& thr : thrExist){
            thrPool->waitExist(thr);
        }
    }
    
    SNEngine::SNEngine(Net& brNet, 
        std::function<void(const std::string&)> sts) : stsCBack_(sts){

        operats_ = brNet.operats;
        nodes_ = brNet.nodes;
        for (auto& n : brNet.nodes)
            ndStates_[n.first] = ndState();

        // только для fwd, поскольку назад может и не пойдем
        thrPoolForward_ = new ThreadPool(bind(&SNEngine::operatorThreadForward, this, std::placeholders::_1));
       
        createThreadsFwd(brNet.nodes, thrPoolForward_);
    }
    
    SNEngine::~SNEngine(){

        fWorkEnd_ = true;
        if (thrPoolForward_) delete thrPoolForward_;
        if (thrPoolBackward_) delete thrPoolBackward_;
    }
            
    /// прямой проход
    bool SNEngine::forward(const SN_Base::operationParam& operPrm){
        
        operParam_ = operPrm;

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
            auto& pn = prevNodes[0];                SN_ENG_DMESS("node " + nname + " actionForward single prevNode " + pn)
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
                    SN_ENG_DMESS("node " + nname + " actionForward multi prevNode " + n)
                    neighb.push_back(operats_[n]);
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
            auto& nd = nodes[nname].nextNodes[0];         SN_ENG_DMESS("node " + nname + " actionBackward single prevNode " + nd)
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
                SN_ENG_DMESS("node " + nname + " actionBackward multi prevNode " + n)
                neighb.push_back(operats_[n]);
            }
            
            /// выполнение оператора
            SN_ENG_DMESS("node " + nname + " actionBackward multi")
            operats_[nname]->Do(operParam_, neighb);
        }
    }
        
    /// сброс готовности старта для след-х узлов
    void SNEngine::resetPreStartNode(std::map<std::string, Node>& nodes, const std::string& nname){
            
        auto getThrNode = [&nodes](const std::string& nname){

            string res = nname;
            while (true){ 

                if (nodes[res].prevNodes.size() != 1) break;

                string pn = nodes[res].prevNodes[0];
                
                if (nodes[pn].nextNodes.size() > 1) break;

                res = pn;
            }

            return res;
        };

        if (nodes[nname].prevNodes.size() == 1)
            thrPoolForward_->resetPrestart(getThrNode(nname));
        else{    
            /// все пред узлы не готовы к запуску?
            bool allPrevNoPrest = true;
            for (auto& n : nodes[nname].prevNodes){
                            
                if (thrPoolForward_->isPrestart(getThrNode(n))){
                    allPrevNoPrest = false;
                    break;
                }
            }

            if (allPrevNoPrest)
                thrPoolForward_->resetPrestart(nname);
        }

        if (thrPoolForward_->isPrestart(getThrNode(nname))) return;
            
        /// для всех след узлов тоже сбрас готовность
        vector<string> nextNodes{ nname };
        while (!nextNodes.empty()){

            string snd = nextNodes.back();
            nextNodes.pop_back();
                            
            if (nodes[snd].nextNodes.empty()) continue;

            for (auto& nn : nodes[snd].nextNodes){

                /// у след узла только один предыд? добавляем в пул для сброса готовности
                if (nodes[nn].prevNodes.size() == 1){
                    thrPoolForward_->resetPrestart(nn);
                    nextNodes.push_back(nn);
                }
                else{

                    /// все пред узлы не готовы к запуску?
                    bool allPrevNoPrest = true;
                    for (auto& n : nodes[nn].prevNodes){
                        if (thrPoolForward_->isPrestart(getThrNode(n)))
                            allPrevNoPrest = false;
                    }

                    /// добавляем в пул для сброса готовности
                    if (allPrevNoPrest){
                        thrPoolForward_->resetPrestart(nn);
                        nextNodes.push_back(nn);
                    }
                }
            }                
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

                resetPreStartNode(nodes, nn);

                /// откат в начало
                selWay = nnameMem;

                /// тек поток тормозим
                thrPoolForward_->finish(nnameMem);        SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
            }            
            /// у след узла один предыдущий?
            else if (nodes[nn].prevNodes.size() == 1){

                /// продолжение пути оставляем в этом же потоке
                selWay = nn;                              SN_ENG_DMESS("node " + nname + " selWay " + nn)
            }
            else{
                /// новый поток для след узла
                if (!thrPoolForward_->isRun(nn)){
                    thrPoolForward_->startTask(nn);       SN_ENG_DMESS("node " + nname + " start thread " + nn)
                }

                /// откат в начало
                selWay = nnameMem;

                /// тек поток тормозим
                thrPoolForward_->finish(nnameMem);        SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
            }
        }
        /// конец пути?
        else if (nextNodes.empty()){

            /// откат в начало
            selWay = nnameMem;

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
                    else
                        resetPreStartNode(nodes, way);
                }
            }
                                            
            /// ответвления расталкиваем по другим потокам
            for (auto& nw : nextWays){
                if (!thrPoolForward_->isRun(nw)){
                    thrPoolForward_->startTask(nw);       SN_ENG_DMESS("node " + nname + " start thread " + nw)
                }
            }

            /// откат в начало
            selWay = nnameMem;

            /// тек поток тормозим
            thrPoolForward_->finish(nnameMem);           SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
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

                /// откат в начало
                selWay = nnameMem;

                /// тек поток тормозим
                thrPoolBackward_->finish(nnameMem);          SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
            }
            /// у пред узла один следующий?
            else if(nodes[pn].nextNodes.size() == 1){

                /// продолжение пути оставляем в этом же потоке
                selWay = pn;                                 SN_ENG_DMESS("node " + nname + " selWay " + pn)
            }
            else{                
                /// новый поток для след узла
                if (!thrPoolBackward_->isRun(pn)){
                    thrPoolBackward_->startTask(pn);         SN_ENG_DMESS("node " + nname + " start thread " + pn)
                }
                /// откат в начало
                selWay = nnameMem;

                /// тек поток тормозим
                thrPoolBackward_->finish(nnameMem);          SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
            }
        }
        /// конец пути?
        else if (prevNodes.empty()){

            /// откат в начало
            selWay = nnameMem;

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
            for (auto& nw : nextWays){
                if (!thrPoolBackward_->isRun(nw)){
                    thrPoolBackward_->startTask(nw);       SN_ENG_DMESS("node " + nname + " start thread " + nw)
                }
            }

            /// откат в начало
            selWay = nnameMem;

            /// тек поток тормозим
            thrPoolBackward_->finish(nnameMem);            SN_ENG_DMESS("node " + nname + " finish thread " + nnameMem)
        }

        return selWay;
    }

    /// рабочий поток для каждой нити операторов
    void SNEngine::operatorThreadForward(std::string nname){
        
        std::string nnameMem = nname;
                
        auto& nodes = nodes_;
                            
        while (!fWorkEnd_){
                        
            /// ждем след итерацию
            thrPoolForward_->waitStart(nnameMem);

            /// обработка текущего узла
            actionForward(nodes, nname);

            /// выбор следующего узла 
            nname = selectNextForward(nodes, nname, nnameMem);                
        }        
    }

    /// рабочий поток для каждой нити операторов
    void SNEngine::operatorThreadBackward(std::string nname){

        std::string nnameMem = nname;

        auto& nodes = nodes_;

        while (!fWorkEnd_){

            /// ждем след итерацию
            thrPoolBackward_->waitStart(nnameMem);

            /// обработка текущего узла
            actionBackward(nodes, nname);

            /// выбор следующего узла 
            nname = selectNextBackward(nodes, nname, nnameMem);
        }
    }
}
