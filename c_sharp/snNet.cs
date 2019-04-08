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

        private List<node> nodes_ = new List<node>();
        private List<uCBack> ucb_ = new List<uCBack>();             
        private void* net_ = null;             
        
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

        ~Net()
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
               
        /// <summary>
        /// add node (layer)
        /// </summary>
        /// <typeparam name="T"> operator type </typeparam>
        /// <param name="name"> name node in architecture of net</param>
        /// <param name="nd"> tensor node</param>
        /// <param name="nextNodes"> next nodes through a space</param>
        /// <returns>ref Net</returns>
        public Net addNode<T>(string name, T nd, string nextNodes){

            nodes_.Add(new node(name, ((IOperator)nd).name(), ((IOperator)nd).getParamsJn(), nextNodes));
            
            return this;
        }

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snSetParamNode(void* net, IntPtr name, IntPtr prms); 
                
        /// <summary>
        /// update param node (layer)
        /// </summary>
        /// <typeparam name="T"> operator type</typeparam>
        /// <param name="name"> name node in architecture of net</param>
        /// <param name="nd"> tensor node</param>
        /// <returns> true ok</returns>
        public bool updateNode<T>(string name, T nd)
        {

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
         
        /// <summary>
        /// forward action
        /// </summary>
        /// <param name="isLern"> is lerning ?</param>
        /// <param name="inTns"> in tensor NCHW(bsz, ch, h, w)</param>
        /// <param name="outTns"> out result tensor</param>
        /// <returns> true - ok</returns>
        public bool forward(bool isLern, Tensor inTns, Tensor outTns)
        {
            if ((net_ == null) && !createNet()) return false;
                       
            return snForward(net_, isLern, inTns.size(), inTns.data(), outTns.size(), outTns.data());
        }

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snBackward(void* net, float lr, snLSize gsz, float* grad); 
         
        /// <summary>
        /// backward action
        /// </summary>
        /// <param name="lr"> lerning rate</param>
        /// <param name="gradTns"> grad error tensor NCHW(bsz, ch, h, w)</param>
        /// <returns> true - ok</returns>
        public bool backward(float lr, Tensor gradTns)
        {

            if ((net_ == null) && !createNet()) return false;

            return snBackward(net_, lr, gradTns.size(), gradTns.data());
        }
           
        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snTraining(void* net, float lr, snLSize insz, float* iLayer,
            snLSize osz, float* outData, float* targetData, float* outAccurate); 
        
        /// <summary>
        /// cycle forward-backward
        /// </summary>
        /// <param name="lr"> lerning rate</param>
        /// <param name="inTns"> in tensor NCHW(bsz, ch, h, w)</param>
        /// <param name="outTns"> out tensor</param>
        /// <param name="targetTns"> target tensor</param>
        /// <param name="outAccurate"> accurate error</param>
        /// <returns> true - ok</returns>
        public bool training(float lr, Tensor inTns, Tensor outTns, Tensor targetTns, ref float outAccurate)
        {

            if ((net_ == null) && !createNet()) return false;

            float accurate = 0;

            bool ok = snTraining(net_, lr, inTns.size(), inTns.data(), 
                outTns.size(), outTns.data(), targetTns.data(), &accurate);

            outAccurate = accurate;

            return ok;
        }

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snSetWeightNode(void* net, IntPtr name, snLSize wsz, float* wData);
                   
        /// <summary>
        /// set weight of node
        /// </summary>
        /// <param name="name"> name node in architecture of net</param>
        /// <param name="weight"> set weight tensor NCHW(bsz, ch, h, w)</param>
        /// <returns> true - ok</returns>
        public bool setWeightNode(string name, Tensor weight)
        {

            if (net_ == null) return false;

            IntPtr cname = Marshal.StringToHGlobalAnsi(name);

            bool ok = snSetWeightNode(net_, cname, weight.size(), weight.data());

            Marshal.FreeHGlobal(cname);

            return ok;
        }

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snGetWeightNode(void* net, IntPtr name, snLSize* wsz, float** wData);

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern void snFreeResources(float* data, char* str);
                
        /// <summary>
        /// get weight of node
        /// </summary>
        /// <param name="name"> name node in architecture of net</param>
        /// <param name="outWeight"> weight tensor NCHW(bsz, ch, h, w)</param>
        /// <returns> true - ok</returns>
        public bool getWeightNode(string name, ref Tensor outWeight)
        {

            if (net_ == null) return false;

            IntPtr cname = Marshal.StringToHGlobalAnsi(name);

            snLSize wsz; float* wdata = null;
            bool ok = snGetWeightNode(net_, cname, &wsz, &wdata);

            Marshal.FreeHGlobal(cname);

            if (ok)
            {
                outWeight = new Tensor(wsz, wdata);

                snFreeResources(wdata, null);
            }

            return ok;
        }
        
        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snGetOutputNode(void* net, IntPtr name, snLSize* wsz, float** wData);
               
        /// <summary>
        /// get output of node
        /// </summary>
        /// <param name="name"> name node in architecture of net</param>
        /// <param name="output"> output tensor NCHW(bsz, ch, h, w)</param>
        /// <returns> true - ok</returns>
        public bool getOutputNode(string name, ref Tensor output)
        {

            if (net_ == null) return false;

            IntPtr cname = Marshal.StringToHGlobalAnsi(name);

            snLSize osz; float* odata = null;
            bool ok = snGetOutputNode(net_, cname, &osz, &odata);

            Marshal.FreeHGlobal(cname);

            if (ok)
            {
                output = new Tensor(osz, odata);

                snFreeResources(odata, null);
            }

            return ok;
        }
        
        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snSaveAllWeightToFile(void* net, IntPtr path);
               
        /// <summary>
        /// save all weight's in file
        /// </summary>
        /// <param name="path"> file path</param>
        /// <returns> true - ok</returns>
        public bool saveAllWeightToFile(string path)
        {

            if (net_ == null) return false;

            IntPtr cpath = Marshal.StringToHGlobalAnsi(path);

            bool ok = snSaveAllWeightToFile(net_, cpath);

            Marshal.FreeHGlobal(cpath);

            return ok;
        }
            
        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snLoadAllWeightFromFile(void* net, IntPtr path); 

        /// <summary>
        /// load all weight's from file
        /// </summary>
        /// <param name="path">file path</param>
        /// <returns>true - ok</returns>
        public bool loadAllWeightFromFile(string path){

            if ((net_ == null) && !createNet()) return false;

            IntPtr cpath = Marshal.StringToHGlobalAnsi(path);

            bool ok = snLoadAllWeightFromFile(net_, cpath); 

            Marshal.FreeHGlobal(cpath);

            return ok;
        }

        ///// add user callback
        ///// @param[in] name - name userCBack in architecture of net
        ///// @param[in] cback - call back function
        ///// @param[in] udata - aux data
        ///// @return true - ok    
        //bool addUserCBack(string name, snUserCBack cback, snUData udata){

        //    bool ok = true;
        //    if (net_)
        //       ok = snAddUserCallBack(net_, name.c_str(), cback, udata);
        //    else
        //        ucb_.push_back(uCBack{ name, cback, udata });

        //    return ok;
        //}

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern bool snGetArchitecNet(void* net, char** arch); 

        /// <summary>
        /// architecture of net in json
        /// </summary>
        /// <returns> jn arch</returns>
        public string getArchitecNetJN(){

            if ((net_ == null) && !createNet()) return "";

            char* arch = null;
            bool ok = snGetArchitecNet(net_, &arch);

            string ret = "";
            if (ok){
                ret = Marshal.PtrToStringAnsi((IntPtr)arch);

                snFreeResources(null, arch);
            }
            return ret;
        }
          
        private bool createNet(){
           
            if (net_ != null) return true;

            if (nodes_.Count == 0) return false;

            string beginNode = nodes_[0].name,
                   prevEndNode = nodes_[nodes_.Count - 1].name;

            foreach(node nd in nodes_){
                if (nd.opr == "Input") beginNode = nd.nextNodes;
                if (nd.nextNodes == "Output"){
                    prevEndNode = nd.name;
                    nd.nextNodes = "EndNet";
                }
            }

            string ss;
            ss = "{" + 
                "\"BeginNet\":" +
                "{" +
                "\"NextNodes\":\"" + beginNode + "\"" +
                "}," +

                "\"Nodes\":" +
                "[";

            int sz = nodes_.Count;
            for (int i = 0; i < sz; ++i){

                node nd = nodes_[i];

                if ((nd.opr == "Input") || (nd.opr == "Output"))
                    continue;
                                
                ss += "{" +
                    "\"NodeName\":\"" + nd.name + "\"," +
                    "\"NextNodes\":\"" + nd.nextNodes + "\"," +
                    "\"OperatorName\":\"" + nd.opr + "\"," +
                    "\"OperatorParams\":" + nd.lparams + "" +
                    "}";

                if (i < sz - 1)  ss += ",";
            }
          
            ss += "]," +

                "\"EndNet\":"  +                        
                "{" +
                "\"PrevNode\":\"" + prevEndNode + "\"" +
                "}" +
                "}";
           
           
            return createNetJN(ss);
        }

        [DllImport("libskynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern void* snCreateNet(IntPtr jnnet, IntPtr isnet); 

        private bool createNetJN(string jnNet){
            
            if (net_ != null) return true;

            IntPtr cerr = Marshal.AllocHGlobal(256);
            IntPtr cnet = Marshal.StringToHGlobalAnsi(jnNet);

            Marshal.WriteByte(cerr, (byte)'\0');
            net_ = snCreateNet(cnet, cerr);

            if (net_ == null) 
              Console.WriteLine(Marshal.PtrToStringAnsi(cerr));

            string rr = Marshal.PtrToStringAnsi(cerr);

            Marshal.FreeHGlobal(cerr);
            Marshal.FreeHGlobal(cnet);

            //if (net_ != null){
            //    for (auto& cb : ucb_)
            //        snAddUserCallBack(net_, cb.name.c_str(), cb.cback, cb.udata);
            //}

            return net_ != null;
        }
    }
}
