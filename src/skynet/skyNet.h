

#ifndef SKYNET_C_API_H_
#define SKYNET_C_API_H_

#ifdef _WIN32
#ifdef SKYNETDLL_EXPORTS
#define SKYNET_API __declspec(dllexport)
#else
#define SKYNET_API __declspec(dllimport)
#endif
#else
#define SKYNET_API
#endif


namespace SN_API{

	extern "C" {

		typedef float snFloat;

		// размер слоя данных
		struct snLSize{

			size_t w, h, ch, bch; ///< ширина, высота, каналы, батч
			snLSize(size_t w_ = 1, size_t h_ = 1, size_t ch_ = 1, size_t bch_ = 1) :
				w(w_), h(h_), ch(ch_), bch(bch_){};
		};

		/// объект нсети
		typedef void* skyNet;

		typedef void* snUData;                                      ///< польз данные	
		typedef void(*snStatusCBack)(const char* mess, snUData);    ///< статус callback

		/// создать объект нсети
		/// @param[in] jnNet - архитектура сети в JSON
		/// @param[out] out_err - ошибка разбора jsNet. "" - ok. Память выделяет польз-ль.
		/// @param[in] statusCBack - callback состояния. Необязательно
		/// @param[in] udata - польз данные (для callback состояния). Необязательно
		SKYNET_API skyNet snCreateNet(const char* jnNet,
			                          char* out_err /*sz 256*/, 
									  snStatusCBack = nullptr, 
			                          snUData = nullptr);

		/// тренинг - цикл вперед-назад с автокоррекцией весов
		/// @param[in] skyNet - объект нсети
		/// @param[in] lr - скорость обучения
		/// @param[in] iLayer - входной слой
		/// @param[in] lsz - размер вх слоя
		/// @param[in] targetData - целевой результат, размер должен соот-ть разметке. Память выделяет польз-ль.
		/// @param[out] outData - выходной результат, размер соот-ет разметке. Память выделяет польз-ль.
		/// @param[in] tsz - размер целевого и выходного результата. Задается для проверки.
		/// @param[out] outAccurate - текущая точность
		/// @return true - ok
		SKYNET_API bool snTraining(skyNet, 
			                       snFloat lr,
								   snFloat* iLayer,
								   snLSize lsz,
								   snFloat* targetData,
								   snFloat* outData,
								   snLSize tsz,
								   snFloat* outAccurate);

		/// прямой проход
		/// @param[in] skyNet - объект нсети
		/// @param[in] isLern - обучение?
		/// @param[in] iLayer - входной слой
		/// @param[in] lsz - размер вх слоя
		/// @param[out] outData - выходной результат, размер соот-ет разметке. Память выделяет польз-ль.
		/// @param[in] osz - размер выходного результата. Задается для проверки.
		/// @return true - ok
		SKYNET_API bool snForward(skyNet,
			                      bool isLern,
			                      snFloat* iLayer,
								  snLSize lsz,
								  snFloat* outData,
								  snLSize osz);

		/// обратный проход
		/// @param[in] skyNet - объект нсети
		/// @param[in] lr - скорость обучения
		/// @param[in] inGradErr - градиент ошибки, размер должен соот-ть выходному результату
		/// @param[in] gsz - размер градиента ошибки. Задается для проверки.
		/// @return true - ok
		SKYNET_API bool snBackward(skyNet,
			                       snFloat lr,
								   snFloat* inGradErr,
								   snLSize gsz);

		
		/// задать веса узла сети
		/// @param[in] skyNet - объект нсети
		/// @param[in] nodeName - имя узла
		/// @param[in] inData - данные
		/// @param[in] dsz - размер данных
		/// @return true - ok
		SKYNET_API bool snSetWeightNode(skyNet,
			                            const char* nodeName,
									    const snFloat* inData,
									    snLSize dsz);

		/// вернуть веса узла сети
		/// @param[in] skyNet - объект нсети
		/// @param[in] nodeName - имя узла
		/// @param[out] outData - данные. Сначала передать NULL, потом передавать его же. 
		/// @param[out] outSz - размер данных
		/// @return true - ok
		SKYNET_API bool snGetWeightNode(skyNet,
			                            const char* nodeName,
									    snFloat** outData,
									    snLSize* outSz);

		/// нормализация слоя по батчу
		struct batchNorm{
			snFloat* mean;      ///< среднее вх значений. Память выделяет польз-ль
			snFloat* varce;     ///< дисперсия вх значений
			snFloat* scale;     ///< коэф γ
			snFloat* schift;    ///< коэф β
		};

		/// задать нормализацию для узла
		/// @param[in] skyNet - объект нсети
		/// @param[in] nodeName - имя узла
		/// @param[in] inData - данные
		/// @param[in] dsz - размер данных
		/// @return true - ok
		SKYNET_API bool snSetBatchNormNode(skyNet,
			                               const char* nodeName,
										   const batchNorm inData,
			                               snLSize dsz);

		/// вернуть нормализацию узла
		/// @param[in] skyNet - объект нсети
		/// @param[in] nodeName - имя узла
		/// @param[out] outData - данные 
		/// @param[out] outSz - размер данных
		/// @return true - ok
		SKYNET_API bool snGetBatchNormNode(skyNet,
			                               const char* nodeName,
			                               batchNorm* outData,
			                               snLSize* outSz);
						
		/// задать входные данные узла (актуально для доп входов)
		/// @param[in] skyNet - объект нсети
		/// @param[in] nodeName - имя узла
		/// @param[in] inData - данные
		/// @param[in] dsz - размер данных
		/// @return true - ok
		SKYNET_API bool snSetInputNode(skyNet,
			                           const char* nodeName,
			                           const snFloat* inData,
			                           snLSize dsz);

		/// вернуть выходные значения узла (актуально для доп выходов)
		/// @param[in] skyNet - объект нсети
		/// @param[in] nodeName - имя узла
		/// @param[out] outData - данные. Сначала передать NULL, потом передавать его же. 
		/// @param[out] outSz - размер данных
		/// @return true - ok
		SKYNET_API bool snGetOutputNode(skyNet,
			                            const char* nodeName,
			                            snFloat** outData,
			                            snLSize* outSz);

		/// задать градиент значения узла (актуально для доп выходов)
		/// @param[in] skyNet - объект нсети
		/// @param[in] nodeName - имя узла
		/// @param[in] inData - данные
		/// @param[in] dsz - размер данных
		/// @return true - ok
		SKYNET_API bool snSetGradientNode(skyNet,
			                              const char* nodeName,
			                              const snFloat* inData,
			                              snLSize dsz);

		/// вернуть градиент значения узла (актуально для доп выходов)
		/// @param[in] skyNet - объект нсети
		/// @param[in] nodeName - имя узла
		/// @param[out] outData - данные. Сначала передать NULL, потом передавать его же. 
		/// @param[out] outSz - размер данных
		/// @return true - ok
		SKYNET_API bool snGetGradientNode(skyNet,
			                              const char* nodeName,
			                              snFloat** outData,
			                              snLSize* outSz);

		/// задать параметры узла
		/// @param[in] skyNet - объект нсети
		/// @param[in] nodeName - имя узла
		/// @param[in] jnParam - параметры узла в JSON. 
		/// @return true - ok
		SKYNET_API bool snSetParamNode(skyNet,
			                           const char* nodeName,
									   const char* jnParam);

		/// вернуть параметры узла
		/// @param[in] skyNet - объект нсети
		/// @param[in] nodeName - имя узла
		/// @param[out] jnParam - параметры узла в JSON. Память выделяет пользователь. 
		/// @return true - ok
		SKYNET_API bool snGetParamNode(skyNet,
			                           const char* nodeName,
									   char* jnParam /*minsz 256*/);

		/// вернуть архитектуру сети
		/// @param[in] skyNet - объект нсети
		/// @param[out] jnNet - архитектура сети в JSON. Память выделяет пользователь.
		/// @return true - ok
		SKYNET_API bool snGetArchitecNet(skyNet,
			                             char* jnNet /*minsz 2048*/);

		/// освободить объект сети
		/// @param[in] skyNet - объект нсети
		SKYNET_API void snFreeNet(skyNet);


	}   // extern "C"
}       // SN_API
#endif  // SKYNET_C_API_H_