
#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>

// потоки на каждой итерации те же самые.
class ThreadPool {
public:

	ThreadPool(std::function<void(std::string node)> func) : func_(func){}

	~ThreadPool(){
		std::lock_guard<std::mutex> lk(mtx_);
		fWorkEnd_ = true;
		
		for (auto& r : ready_){
			r.second->end();
		}

		for (auto& thr : threads_){
			if (thr.second->joinable())
				thr.second->join();
			delete thr.second;
		}
		
		for (auto& r : ready_)
			delete r.second;
	}

	void addNode(const std::string& node){
		std::lock_guard<std::mutex> lk(mtx_);
			
		if (fWorkEnd_) return;

		ready_[node] = new SReady();
		threads_[node] = new std::thread(func_, node);	
	}
	
	void startTask(const std::string& node){
		std::lock_guard<std::mutex> lk(mtx_);

		if (fWorkEnd_) return;
			
		ready_[node]->start();       	
     }
	
	void finish(const std::string& node){
		 std::lock_guard<std::mutex> lk(mtx_);

		 ready_[node]->finish();
	 }

	void waitAll(){
		if (fWorkEnd_) return;

		for (auto& r : ready_){
		    r.second->waitFinish();
		}
	 }

	void waitStart(const std::string& node){
		if (fWorkEnd_) return;

		ready_[node]->exist();
		
		ready_[node]->waitStart();
	 }

	void waitFinish(const std::string& node){
		if (fWorkEnd_) return;
		
		ready_[node]->waitFinish();
	 }

	void waitExist(const std::string& node){
		if (fWorkEnd_) return;

		ready_[node]->waitExist();
	}

	bool isRun(const std::string& node){
		std::lock_guard<std::mutex> lk(mtx_);

		bool isRn = ready_[node]->isRun() || !ready_[node]->isPrestart();
	
		return isRn;
	}

	bool isPrestart(const std::string& node){
		std::lock_guard<std::mutex> lk(mtx_);
				
		bool isPr = (ready_.find(node) != ready_.end()) && ready_[node]->isPrestart();

		return isPr;
	}

	void preStartAll(){
		std::lock_guard<std::mutex> lk(mtx_);

		for (auto& r : ready_){
			r.second->setPrestart();
		}
	}

	void resetPrestart(const std::string& node){
		std::lock_guard<std::mutex> lk(mtx_);

		if (ready_.find(node) != ready_.end())
		   ready_[node]->resetPrestart();
	}
	 	 	 	 	
 private:

	 class SReady{
	 public:
		 SReady(){}
		 ~SReady(){
			 end();
		 }

		 void waitStart() {
			 std::unique_lock<std::mutex> lk(lkStart_);
			 if (!end_ && !run_){
				 cvrStart_.wait(lk);
			 }
		 }

		 void waitFinish() {
			 std::unique_lock<std::mutex> lk(lkFinish_);
			 if (!end_ && (run_ || preStart_)){
				 cvrFinish_.wait(lk);
			 }
		 }
		 
		 void waitExist() {
			 std::unique_lock<std::mutex> lk(lkExist_);
			 if (!end_ && !isExist_){
				 cvrExist_.wait(lk);
			 }
		 }

		 void start(){
			 std::lock_guard<std::mutex> lk(lkStart_);
			 if (!run_){
				 run_ = true;
				 cvrStart_.notify_all();
			 }
		 }

		 void finish() {
			 std::lock_guard<std::mutex> lk(lkFinish_);
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
			 std::lock_guard<std::mutex> lk(lkExist_);
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
		 }

		 bool isRun(){

			 return run_;
		 }

		 bool isPrestart(){

			 return preStart_;
		 }

	 private:
		 std::mutex  lkStart_, lkFinish_, lkExist_;
		 std::condition_variable cvrStart_, cvrFinish_, cvrExist_;
		 bool run_ = false, preStart_ = false, isExist_ = false, end_ = false;
	 };

	 std::function<void(std::string node)> func_ = nullptr;

	 std::mutex mtx_;
	 std::map<std::string, std::thread*> threads_;
	 std::map<std::string, SReady*> ready_;
	 bool fWorkEnd_ = false;

};