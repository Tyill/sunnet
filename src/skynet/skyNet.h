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

        /// data layer size
        struct snLSize{

            size_t w, h, ch, bch; ///< width, height, channels, batch
            snLSize(size_t w_ = 1, size_t h_ = 1, size_t ch_ = 1, size_t bch_ = 1) :
                w(w_), h(h_), ch(ch_), bch(bch_){};
        };

        /// object net
        typedef void* skyNet;

        typedef void* snUData;                                      ///< user data    
        typedef void(*snStatusCBack)(const char* mess, snUData);    ///< status callback

        /// create net
        /// @param[in] jnNet - network architecture in JSON
        /// @param[out] out_err - parse error jnNet. "" - ok. The memory is allocated by the user
        /// @param[in] statusCBack - callback state. Not necessary
        /// @param[in] udata - user data. Not necessary
        SKYNET_API skyNet snCreateNet(const char* jnNet,
                                      char* out_err /*sz 256*/, 
                                      snStatusCBack = nullptr, 
                                      snUData = nullptr);

        /// training - a cycle forward-back with auto-correction of weights
        /// @param[in] skyNet - object net
        /// @param[in] lr - learning rate
        /// @param[in] iLayer - input layer
        /// @param[in] lsz - input layer size
        /// @param[in] targetData - target, the size must match the markup. The memory is allocated by the user
        /// @param[out] outData - result, the size must match the markup. The memory is allocated by the user
        /// @param[in] tsz - size of target and result. Sets for verification
        /// @param[out] outAccurate - current accuracy
        /// @return true - ok
        SKYNET_API bool snTraining(skyNet, 
                                   snFloat lr,
                                   snFloat* iLayer,
                                   snLSize lsz,
                                   snFloat* targetData,
                                   snFloat* outData,
                                   snLSize tsz,
                                   snFloat* outAccurate);

        /// forward pass
        /// @param[in] skyNet - object net
        /// @param[in] isLern - is lern?
        /// @param[in] iLayer - input layer
        /// @param[in] lsz - input layer size
        /// @param[out] outData - result, the size must match the markup. The memory is allocated by the user
        /// @param[in] osz - size of result. Sets for verification
        /// @return true - ok
        SKYNET_API bool snForward(skyNet,
                                  bool isLern,
                                  snFloat* iLayer,
                                  snLSize lsz,
                                  snFloat* outData,
                                  snLSize osz);

        /// backward pass
        /// @param[in] skyNet - object net
        /// @param[in] lr - learning rate
        /// @param[in] inGradErr - error gradient, the size must match the output
        /// @param[in] gsz - size of the error gradient. Sets for verification
        /// @return true - ok
        SKYNET_API bool snBackward(skyNet,
                                   snFloat lr,
                                   snFloat* inGradErr,
                                   snLSize gsz);

        
        /// set weight of node
        /// @param[in] skyNet - object net
        /// @param[in] nodeName - name node
        /// @param[in] inData - data
        /// @param[in] dsz - data size
        /// @return true - ok
        SKYNET_API bool snSetWeightNode(skyNet,
                                        const char* nodeName,
                                        const snFloat* inData,
                                        snLSize dsz);

        /// get weight of node
        /// @param[in] skyNet - object net
        /// @param[in] nodeName - name node
        /// @param[out] outData - output data. First pass NULL, then pass it to the same 
        /// @param[out] outSz - output size
        /// @return true - ok
        SKYNET_API bool snGetWeightNode(skyNet,
                                        const char* nodeName,
                                        snFloat** outData,
                                        snLSize* outSz);

        /// batchNorm
        struct batchNorm{
            snFloat* mean;      ///< mean. The memory is allocated by the user
            snFloat* varce;     ///< disp
            snFloat* scale;     ///< coef γ
            snFloat* schift;    ///< coef β
        };

        /// set batchNorm of node
        /// @param[in] skyNet - object net
        /// @param[in] nodeName - name node
        /// @param[in] inData - data
        /// @param[in] dsz - data size
        /// @return true - ok
        SKYNET_API bool snSetBatchNormNode(skyNet,
                                           const char* nodeName,
                                           const batchNorm inData,
                                           snLSize dsz);

        /// get batchNorm of node
        /// @param[in] skyNet - object net
        /// @param[in] nodeName - name node
        /// @param[out] outData - data 
        /// @param[out] outSz - data size
        /// @return true - ok
        SKYNET_API bool snGetBatchNormNode(skyNet,
                                           const char* nodeName,
                                           batchNorm* outData,
                                           snLSize* outSz);
                        
        /// set input node (relevant for additional inputs)
        /// @param[in] skyNet - object net
        /// @param[in] nodeName - name node
        /// @param[in] inData - data
        /// @param[in] dsz - data size
        /// @return true - ok
        SKYNET_API bool snSetInputNode(skyNet,
                                       const char* nodeName,
                                       const snFloat* inData,
                                       snLSize dsz);

        /// get output node (relevant for additional inputs)
        /// @param[in] skyNet - object net
        /// @param[in] nodeName - name node
        /// @param[out] outData - data. First pass NULL, then pass it to the same 
        /// @param[out] outSz - data size
        /// @return true - ok
        SKYNET_API bool snGetOutputNode(skyNet,
                                        const char* nodeName,
                                        snFloat** outData,
                                        snLSize* outSz);

        /// set gradient node (relevant for additional outputs)
        /// @param[in] skyNet - object net
        /// @param[in] nodeName - name node
        /// @param[in] inData - data
        /// @param[in] dsz - data size
        /// @return true - ok
        SKYNET_API bool snSetGradientNode(skyNet,
                                          const char* nodeName,
                                          const snFloat* inData,
                                          snLSize dsz);

        /// get gradient node (relevant for additional outputs)
        /// @param[in] skyNet - object net
        /// @param[in] nodeName - name node
        /// @param[out] outData - data. First pass NULL, then pass it to the same 
        /// @param[out] outSz - data size
        /// @return true - ok
        SKYNET_API bool snGetGradientNode(skyNet,
                                          const char* nodeName,
                                          snFloat** outData,
                                          snLSize* outSz);

        /// set params of node
        /// @param[in] skyNet - object net
        /// @param[in] nodeName - name node
        /// @param[in] jnParam - params of node in JSON. 
        /// @return true - ok
        SKYNET_API bool snSetParamNode(skyNet,
                                       const char* nodeName,
                                       const char* jnParam);

        /// get params of node
        /// @param[in] skyNet - object net
        /// @param[in] nodeName - name node
        /// @param[out] jnParam - params of node in JSON. The memory is allocated by the user 
        /// @return true - ok
        SKYNET_API bool snGetParamNode(skyNet,
                                       const char* nodeName,
                                       char* jnParam /*minsz 256*/);

        /// get architecture of net
        /// @param[in] skyNet - object net
        /// @param[out] jnNet - architecture of net in JSON. The memory is allocated by the user
        /// @return true - ok
        SKYNET_API bool snGetArchitecNet(skyNet,
                                         char* jnNet /*minsz 2048*/);

        /// free object net
        /// @param[in] skyNet - object net
        SKYNET_API void snFreeNet(skyNet);


    }   // extern "C"
}       // SN_API
#endif  // SKYNET_C_API_H_
