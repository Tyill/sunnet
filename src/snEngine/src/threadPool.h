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
#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>

// потоки на каждой итерации те же самые.
class ThreadPool {
public:

    ThreadPool(std::function<void(std::string node)> func) : func_(func){}

    ~ThreadPool(){
        fWorkEnd_ = true;
        
        for (auto& r : ready_){
            r.second.end();
        }

        for (auto& thr : threads_){
            if (thr.second.joinable())
                thr.second.join();
        }
    }

    void addThread(const std::string& node){
        std::lock_guard<std::mutex> lk(mtx_);
            
        if (fWorkEnd_) return;
        
        threads_[node] = std::thread(func_, node);    
    }

    void addNode(const std::string& node){
        std::lock_guard<std::mutex> lk(mtx_);

        if (fWorkEnd_) return;

        ready_[node];
    }
    
    void startTask(const std::string& node){
        std::lock_guard<std::mutex> lk(mtx_);

        if (fWorkEnd_) return;
               
        if (ready_[node].isRun() || !ready_[node].isPrestart()) return;

        for (auto& thr : threads_){
            if (!ready_[thr.first].isRun()){

                ready_[thr.first].start(node);
                ready_[node].start(node);

                return;
            }
        }
            
        threads_[node] = std::thread(func_, node);

        ready_[node].exist();
        ready_[node].start(node);
    }

    void restartTask(const std::string& thr, const std::string& node){
        std::lock_guard<std::mutex> lk(mtx_);

        if (fWorkEnd_) return;

        auto workNode = ready_[thr].getWorkNode();
        if (workNode != thr)
           ready_[workNode].finish();
       
        ready_[thr].finish();

        if (!ready_[node].isRun() && ready_[node].isPrestart()){
            ready_[thr].start(node);
            ready_[node].start(node);
        }
    }
    
    void finish(const std::string& node){
         std::lock_guard<std::mutex> lk(mtx_);

         auto workNode = ready_[node].getWorkNode();
         if (!workNode.empty() && (workNode != node))
             ready_[workNode].finish();

         ready_[node].finish();
     }

    void waitAll(){
        if (fWorkEnd_) return;

        for (auto& r : ready_){
            r.second.waitFinish();
        }
     }

    std::string waitStart(const std::string& node){
        if (fWorkEnd_) return node;

        if (!ready_[node].isExist())
          ready_[node].exist();
        
        return ready_[node].waitStart();
     }

    void waitFinish(const std::string& node){
        if (fWorkEnd_) return;

        ready_[node].waitFinish();
     }

    void waitExist(const std::string& node){
        if (fWorkEnd_) return;

        ready_[node].waitExist();
    }
       
    bool isPrestart(const std::string& node){
        std::lock_guard<std::mutex> lk(mtx_);
                
        bool isPr = (ready_.find(node) != ready_.end()) && ready_[node].isPrestart();

        return isPr;
    }

    void preStartAll(){
        std::lock_guard<std::mutex> lk(mtx_);

        for (auto& r : ready_){
            r.second.setPrestart();
        }
    }

    void resetPrestart(const std::string& node){
        std::lock_guard<std::mutex> lk(mtx_);

        if (ready_.find(node) != ready_.end())
           ready_[node].resetPrestart();
    }
                        
 private:

     class SReady{
     public:
         SReady(){}
         ~SReady(){
             end();
         }
     
         std::string waitStart() {
             std::unique_lock<std::mutex> lk(lkStart_);
             if (!end_ && !run_){
                 cvrStart_.wait(lk);
             }
             return workNode_;
         }

         void waitFinish() {
             std::unique_lock<std::mutex> lk(lkFinish_);
             if (!end_ && preStart_){
                 cvrFinish_.wait(lk);
             }
         }
         
         void waitExist() {
             std::unique_lock<std::mutex> lk(lkExist_);
             if (!end_ && !isExist_){
                 cvrExist_.wait(lk);
             }
         }

         void start(const std::string& workNode){
             workNode_ = workNode;
             if (!run_){                 
                 run_ = true;
                 cvrStart_.notify_all();
             }
         }

         void finish() {
             if (run_){
                 run_ = false;
                 preStart_ = false;
                 cvrFinish_.notify_all();
             }
         }
                  
         void end(){
             end_ = true;
             cvrStart_.notify_all();
             cvrFinish_.notify_all();
             cvrExist_.notify_all();
         }

         void exist(){
             if (!isExist_){
                 isExist_ = true;
                 cvrExist_.notify_all();
             }
         }

         void setPrestart(){

             preStart_ = true;
         }

         void resetPrestart(){

             preStart_ = false;
             cvrFinish_.notify_all();
         }

         bool isExist(){

             return isExist_;
         }

         bool isRun(){

             return run_;
         }

         bool isPrestart(){

             return preStart_;
         }

         std::string getWorkNode(){

             return workNode_;
         }

     private:
         std::string workNode_;
         std::mutex lkStart_, lkFinish_, lkExist_;
         std::condition_variable cvrStart_, cvrFinish_, cvrExist_;
         bool run_ = false, preStart_ = false, isExist_ = false, end_ = false;
     };

     std::function<void(std::string node)> func_ = nullptr;

     std::mutex mtx_;
     std::map<std::string, std::thread> threads_;
     std::map<std::string, SReady> ready_;
     bool fWorkEnd_ = false;

};