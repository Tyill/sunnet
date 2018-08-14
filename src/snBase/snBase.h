
#pragma once

#include <cstdint>
#include <cassert>
#include <vector>
#include <map>

#define SN_DEBUG

#ifdef SN_DEBUG
#define SN_PRINTMESS(mess) printf("%s \n", (mess).c_str());
#else
#define SN_PRINTMESS(mess);
#endif


namespace SN_Base{
	
	typedef float snFloat;
	
	/// режим работы сети - прямой/обратный проход
	enum snAction{
		forward = 0,
		backward = 1,
	};
		    
	/// размер
	struct snSize{
		size_t w, h, d, n, p;
				
		snSize(size_t w_ = 1, size_t h_ = 1, size_t d_ = 1, size_t n_ = 1, size_t p_ = 1) :
			w(w_), h(h_), d(d_), n(n_), p(p_){}

		size_t size() const{
			return w * h * d * n * p;
		}
				
		friend bool operator==(const snSize& left, const snSize& right){

			return (left.w == right.w) && (left.h == right.h) && (left.d == right.d) 
				&& (left.n == right.n) && (left.p == right.p);
		}

		friend bool operator!=(const snSize& left, const snSize& right){

			return (left.w != right.w) || (left.h != right.h) || (left.d != right.d)
				|| (left.n != right.n) || (left.p != right.p);
		}

	};
		
	/// тензор - вход данные и выходные данные каждого узла сети.
	struct Tensor{
				
		explicit Tensor(const snSize& sz = snSize(0,0,0,0,0)) : sz_(sz){

			size_t ssz = sz.size();
		
			if (ssz > 0)
				data_ = (snFloat*)calloc(ssz, sizeof(snFloat));
		}

		~Tensor(){			
			if (data_) free(data_);
		}
		
		friend bool operator==(const Tensor& left, const Tensor& right){
						
			return left.sz_ == right.sz_;
		}

		friend bool operator!=(const Tensor& left, const Tensor& right){

			return left.sz_ != right.sz_;
		}

		Tensor& operator=(const Tensor& other){

			setData(other.getData(), other.size());

			return *this;
		}
		
		Tensor& operator+=(const Tensor& other){

			assert(other == *this);

			auto od = other.getData();

			size_t sz = this->size().size();
			for (size_t i = 0; i < sz; ++i){
				data_[i] += od[i];
			}
			
			return *this;
		}

		snFloat* getData() const{
				
			return data_;
		}
				
		void setData(snFloat* data, const snSize& nsz){

			size_t nnsz = nsz.size();
			assert(data && (nnsz > 0));
			
			if (sz_.size() < nnsz)
				data_ = (snFloat*)realloc(data_, nnsz * sizeof(snFloat));
		
			memcpy(data_, data, nnsz * sizeof(snFloat));
			sz_ = nsz;
		}

		void resize(const snSize& nsz){

			size_t nnsz = nsz.size();
			assert(nnsz > 0);

			if (sz_.size() < nnsz)
				data_ = (snFloat*)realloc(data_, nnsz * sizeof(snFloat));
			
			sz_ = nsz;
		}
				
		snSize size() const{

			return sz_;
		}

	private:
		snFloat* data_ = nullptr;

		snSize sz_;
	};

	/// параметры обучения
	struct learningParam{

		bool isAutoCalcError; ///< расчит ошибку автоматич
		bool isLerning;       ///< обучение
		snAction action;      ///< режим работы
		snFloat lr;           ///< коэф скорости обучения
		
		learningParam(bool isAutoCalcError_ = false, bool isLerning_ = false, snAction action_ = snAction::forward, SN_Base::snFloat lr_ = 0.001) :
			isAutoCalcError(isAutoCalcError_), isLerning(isLerning_), action(action_), lr(lr_){}
	};
	
	/// нормализация слоя по батчу
	struct batchNorm{
		std::vector<SN_Base::snFloat> mean;       ///< среднее вх значений
		std::vector<SN_Base::snFloat> varce;      ///< дисперсия вх значений
		std::vector<SN_Base::snFloat> scale;      ///< коэф γ
		std::vector<SN_Base::snFloat> schift;     ///< коэф β
		snSize sz;

		void set(SN_Base::snFloat* mean_,
			     SN_Base::snFloat* varce_,
			     SN_Base::snFloat* scale_,
			     SN_Base::snFloat* schift_,
				 snSize sz_){

			size_t lsz = sz_.w *  sz_.h *  sz_.d;

			mean.resize(lsz);   memcpy(mean.data(), mean_, lsz * sizeof(snFloat));
			varce.resize(lsz);  memcpy(varce.data(), varce_, lsz * sizeof(snFloat));
			scale.resize(lsz);  memcpy(scale.data(), scale_, lsz * sizeof(snFloat));
			schift.resize(lsz); memcpy(schift.data(), schift_, lsz * sizeof(snFloat));

			sz = sz_;
		}

	
	};

	/// базовый оператор сети. Реализация слоев, расчет весов, градиентов, активации и тд.
	/// Все расчетные операторы наследуются от него. 
	class OperatorBase{

	protected:
		/// создать ф-ю
		/// @param name - имя оператора - конкретный класс реализации 
		/// @param node - название узла в символьной структуре НС
		/// @param prms - параметры ф-и. Ключ - имя параметра. (задает польз-ль когда создает сеть - JSON структуру сети).
		OperatorBase(const std::string& name, const std::string& node, std::map<std::string, std::string>& prms) :
			name_(name), node_(node), basePrms_(prms){}
		~OperatorBase(){		
			if (baseInput_) delete baseInput_;
			if (baseWeight_) delete baseWeight_;
			if (baseGrad_) delete baseGrad_;
			if (baseOut_) delete baseOut_;
		}
	public:
					
		/// задать параметры
		virtual bool setInternPrm(std::map<std::string, std::string>& prms){
			basePrms_ = prms;
			return true;
		}

		/// задать входные данные для расчета
		virtual bool setInput(SN_Base::Tensor* in){			
			if (baseInput_) baseInput_->~Tensor();
			baseInput_ = in;
			return true;
		}

		/// задать градиент
		virtual bool setGradient(SN_Base::Tensor* grad){
			if (baseGrad_) baseGrad_->~Tensor();
			baseGrad_ = grad;
			return true;
		}
		
		/// задать веса
		virtual bool setWeight(SN_Base::Tensor* weight){
			if (baseWeight_) baseWeight_->~Tensor();
			baseWeight_ = weight;
			return true;
		}

		/// задать нормализацию
		virtual bool setBatchNorm(const batchNorm& bn){
			baseBatchNorm_ = bn;
			return true;
		}

		/// вернуть параметры
		virtual std::map<std::string, std::string> getInternPrm() const final{
			return basePrms_;
		}

		/// вернуть веса
		virtual SN_Base::Tensor* getWeight() const final{
			return baseWeight_;
		}

		/// вернуть нормализацию
		virtual batchNorm getBatchNorm() const final{
			return baseBatchNorm_;
		}

		/// вернуть выходные данные ф-и
		virtual SN_Base::Tensor* getOutput() const final{
			return baseOut_;
		}

		/// вернуть расчитан градиенты ф-и
		virtual SN_Base::Tensor* getGradient() const final{
			return baseGrad_;
		}

		/// название узла в символьной структуре
		virtual std::string node() const final{
			return node_;
		}

		/// имя оператора - конкретный класс реализации 
		virtual std::string name() const final{
			return name_;
		}

		/// выполнить расчет
		/// @param learnPrm - параметры обучения на тек итерации
		/// @param neighbOpr - соседние операторы, передающие сюда данные
		/// @return - список след узлов куда идти, если след-х > 1. Если ничего не выбрано идем на все
		virtual std::vector<std::string> Do(const learningParam& learnPrm, const std::vector<OperatorBase*>& neighbOpr) = 0;
				
	protected:
		std::string node_;                            ///< имя узла, в котором вычисляется оператор
		std::string name_;                            ///< имя оператора - конкретный класс реализации 
		std::map<std::string, std::string> basePrms_; ///< параметры
		SN_Base::Tensor* baseInput_ = nullptr;        ///< входные данные оператора
		SN_Base::Tensor* baseWeight_ = nullptr;       ///< веса
		SN_Base::Tensor* baseGrad_ = nullptr;         ///< градиенты
		SN_Base::Tensor* baseOut_ = nullptr;          ///< выходные данные оператора
		batchNorm baseBatchNorm_;                     ///< нормализация
	};

	/// узел в символьной структуре НС 
	struct Node{

		std::string name;                           ///< название узла - дбыть уникальным в пределах ветки, без ' ' и '-'. "Begin", "End", "TCP" зарезервированы как начало, конец сети и передача в др ветвь.			                                                                                             
		std::string oprName;                        ///< оператор узла, который выполняется в узле. !Один оператор на узел. Дбыть реализован в FNOperator.dll
		std::map<std::string, std::string> oprPrms; ///< параметры оператора (задает польз-ль при создании ветви)
		std::vector<std::string> prevNodes;         ///< предыдущие узлы (множест число, тк узел мбыть собирательным из неск веток)
		std::vector<std::string> nextNodes;     	///< все возможные след узлы (множест число, тк мбыть разделение на неск параллель нитей), название след узлов (через пробел) возвращает ф-я оператора узла на тек итерации. Если ничего не вернула, идет на все
	};
	
	/// символьная стр-ре НС - последовательный граф операций (мбыть ветвления).
	struct Net{			
		std::map<std::string, Node> nodes;            ///< общая коллекция узлов НС. Ключ - название узла
		std::map<std::string, OperatorBase*> operats; ///< общая коллекция операторов НС. Ключ - название узла
	};

	
	
};