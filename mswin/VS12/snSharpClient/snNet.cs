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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace SN_API
{
    public unsafe class Net
    {
        private class node{
            public string name;
            public string opr;
            public string lparams;
            public string nextNodes;

            public node(string name_, string opr_,  string lparams_, string nextNodes_){

                name = name_;
                opr = opr_;
                lparams = lparams_;
                nextNodes = nextNodes_;
            }
        };

        private class uCBack
        {
            public string name;
            public IntPtr cback;
            public IntPtr udata;           
        };   

        private List<node> nodes_;
        private List<uCBack> ucb_;

        private void* net_ = null;

        private string netStruct_;                 
        
        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern void snVersionLib(IntPtr ver); 
       
        /// <summary>
        /// version library
        /// </summary>
        /// <returns> version </returns>
        public string versionLib()
        {            
            IntPtr cptr = Marshal.AllocHGlobal(16);
            snVersionLib(cptr);

            string ver = Marshal.PtrToStringAnsi(cptr);

            Marshal.FreeHGlobal(cptr);

            return ver;
        }     
               
        /// <summary>
        /// create net
        /// </summary>
        /// <param name="jnNet"> network architecture in JSON </param>
        /// <param name="weightPath"> path to file with weight </param>
        public Net(string jnNet = "", string weightPath = ""){
        
            if (jnNet.Length > 0)
                createNetJN(jnNet);

            if ((net_ != null) && (weightPath.Length > 0))
                loadAllWeightFromFile(weightPath);            
        }

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern void snFreeNet(void* net); 

        public ~Net()
        {
            if (net_ != null)
                snFreeNet(net_);        
        }

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern void snGetLastErrorStr(void* net, IntPtr err); 

        /// <summary>
        /// last error
        /// </summary>
        /// <returns>"" ok</returns>
        public string getLastErrorStr(){

            string err = "";
            if (net_ != null){

                IntPtr cptr = Marshal.AllocHGlobal(256);

                snGetLastErrorStr(net_, cptr);
                err = Marshal.PtrToStringAnsi(cptr);

                Marshal.FreeHGlobal(cptr);
            }

            return err;
        }

        /// add node (layer)
        /// @param[in] name - name node in architecture of net
        /// @param[in] nd - tensor node
        /// @param[in] nextNodes - next nodes through a space
        /// @return ref Net
        public Net addNode<T>(string name, T nd, string nextNodes){

            nodes_.Add(new node(name, ((IOperator)nd).name(), ((IOperator)nd).getParamsJn(), nextNodes));
            
            return this;
        }

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snSetParamNode(void* net, IntPtr prms, IntPtr prms); 

        /// update param node (layer)
        /// @param[in] name - name node in architecture of net
        /// @param[in] nd - tensor node
        /// @return true - ok
        bool updateNode<T>(string name, T nd){

            bool ok = false;
            if (net_ != null){
                              
                IntPtr cname = Marshal.StringToHGlobalAnsi(name);
                IntPtr cprm = Marshal.StringToHGlobalAnsi(((IOperator)nd).getParamsJn());

                ok = snSetParamNode(net_, cname, cprm);
               
                Marshal.FreeHGlobal(cname);
                Marshal.FreeHGlobal(cprm);
            }
            else{
                for (int i = 0; i < nodes_.Count; ++i)
                {
                    if (nodes_[i].name == name)
                    {
                        nodes_[i].lparams = ((IOperator)nd).getParamsJn();
                        ok = true;
                        break;
                    }
                }
            }

            return ok;
        }

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snForward(void* net, bool isLern, snLSize isz, float* iLayer, snLSize osz, float* outData); 
                
        /// forward action
        /// @param[in] isLern - is lerning ?
        /// @param[in] inTns - in tensor
        /// @param[inout] outTns - out result tensor
        /// @return true - ok
        bool forward(bool isLern, Tensor inTns, Tensor outTns){

            if ((net_ == null) && !createNet()) return false;
                       
            return snForward(net_, isLern, inTns.size(), inTns.data(), outTns.size(), outTns.data());
        }

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snBackward(void* net, float lr, snLSize gsz, float* grad); 
      
    
        /// backward action
        /// @param[in] lr - lerning rate
        /// @param[in] gradTns - grad error tensor
        /// @return true - ok
        bool backward(float lr, Tensor gradTns){

            if ((net_ == null) && !createNet()) return false;

            return snBackward(net_, lr, gradTns.size(), gradTns.data());
        }

        ///// training action - cycle forward-backward
        ///// @param[in] lr - lerning rate
        ///// @param[in] inTns - in tensor
        ///// @param[inout] outTns - out tensor
        ///// @param[in] targetTns - target tensor
        ///// @param[inout] outAccurate - accurate error
        ///// @return true - ok
        //bool training(snFloat lr, Tensor& inTns, Tensor& outTns, Tensor& targetTns, snFloat& outAccurate){

        //    if (!net_ && !createNet()) return false;

        //    return snTraining(net_, lr, inTns.size(), inTns.data(), 
        //        outTns.size(), outTns.data(),
        //        targetTns.data(), &outAccurate);
        //}
        
        ///// set weight of node
        ///// @param[in] name - name node in architecture of net
        ///// @param[in] weight - set weight tensor
        ///// @return true - ok
        //bool setWeightNode(const std::string& name, Tensor& weight){

        //    if (!net_) return false;

        //    return snSetWeightNode(net_, name.c_str(), weight.size(), weight.data());
        //}

        ///// get weight of node
        ///// @param[in] name - name node in architecture of net
        ///// @param[out] outWeight - weight tensor
        ///// @return true - ok
        //bool getWeightNode(const std::string& name, Tensor& outWeight){

        //    if (!net_) return false;

        //    snLSize wsz; snFloat* wdata = nullptr;
        //    if (snGetWeightNode(net_, name.c_str(), &wsz, &wdata) && wdata){

        //        outWeight = Tensor(wsz, wdata);

        //        snFreeResources(wdata, 0);
        //        return true;
        //    }
        //    else
        //        return false;
        //}
        
        ///// get output of node
        ///// @param[in] name - name node in architecture of net
        ///// @param[out] output - output tensor
        ///// @return true - ok
        //bool getOutputNode(const std::string& name, Tensor& output){

        //    if (!net_) return false;

        //    snLSize osz; snFloat* odata = nullptr;
        //    if (snGetOutputNode(net_, name.c_str(), &osz, &odata) && odata){

        //        output = Tensor(osz, odata);

        //        snFreeResources(odata, 0);
        //        return true;
        //    }
        //    else
        //        return false;
        //}

        ///// save all weight's in file
        ///// @param[in] path - file path
        ///// @return true - ok
        //bool saveAllWeightToFile(const std::string& path){

        //    if (!net_) return false;

        //    return snSaveAllWeightToFile(net_, path.c_str());
        //}
            
        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern void snLoadAllWeightFromFile(IntPtr net, IntPtr err); 

        /// <summary>
        /// load all weight's from file
        /// </summary>
        /// <param name="path">file path</param>
        /// <returns>true - ok</returns>
        public bool loadAllWeightFromFile(string path){

            if ((net_ == null) && !createNet()) return false;

            return true;//snLoadAllWeightFromFile(net_, net_);
        }

        ///// add user callback
        ///// @param[in] name - name userCBack in architecture of net
        ///// @param[in] cback - call back function
        ///// @param[in] udata - aux data
        ///// @return true - ok
        //bool addUserCBack(const std::string& name, snUserCBack cback, snUData udata){

        //    bool ok = true;
        //    if (net_)
        //       ok = snAddUserCallBack(net_, name.c_str(), cback, udata);
        //    else
        //        ucb_.push_back(uCBack{ name, cback, udata });

        //    return ok;
        //}

        ///// architecture of net in json
        ///// @return jn arch
        //string getArchitecNetJN(){

        //    if (!net_ && !createNet()) return "";

        //    char* arch = nullptr;
        //    snGetArchitecNet(net_, &arch);
            
        //    std::string ret = arch;

        //    snFreeResources(0, arch);

        //    return ret;
        //}
    
      
        private bool createNet(){
           
            return true;
            //if (net_) return true;

            //if (nodes_.empty()) return false;

            //std::string beginNode = nodes_.front().name,
            //            prevEndNode = nodes_.back().name;

            //for (auto& nd : nodes_){
            //    if (nd.opr == "Input") beginNode = nd.nextNodes;
            //    if (nd.nextNodes == "Output"){
            //        prevEndNode = nd.name;
            //        nd.nextNodes = "EndNet";
            //    }
            //}

            //std::stringstream ss;
            //ss << "{"
            //    "\"BeginNet\":"
            //    "{"
            //    "\"NextNodes\":\"" + beginNode + "\""
            //    "},"

            //    "\"Nodes\":"
            //    "[";

            //size_t sz = nodes_.size();
            //for (int i = 0; i < sz; ++i){

            //    auto& nd = nodes_[i];

            //    if ((nd.opr == "Input") || (nd.opr == "Output"))
            //        continue;
                                
            //    ss << "{"
            //        "\"NodeName\":\"" + nd.name + "\","
            //        "\"NextNodes\":\"" + nd.nextNodes + "\","
            //        "\"OperatorName\":\"" + nd.opr + "\","
            //        "\"OperatorParams\":" + nd.params + ""
            //        "}";

            //    if (i < sz - 1)  ss << ",";
            //}
          
            //ss << "],"

            //    "\"EndNet\":"                         
            //    "{"
            //    "\"PrevNode\":\"" + prevEndNode + "\""
            //    "}"
            //    "}";
           
           
            //return createNet(ss.str().c_str());
        }

        private bool createNetJN(string jnNet){
            
            return true;
            //if (net_) return true;

            //char err[256]; err[0] = '\0';
            //net_ = snCreateNet(jnNet.c_str(), err);

            //err_ = err;

            //if (net_){
            //    for (auto& cb : ucb_)
            //        snAddUserCallBack(net_, cb.name.c_str(), cb.cback, cb.udata);
            //}

            //return net_ != nullptr;
        }
    }
}
