using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace SN_API
{
    public class Net
    {
        [DllImport("skynet.dll", CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
        static extern void snVersionLib([MarshalAs(UnmanagedType.LPStr)]string ver); 
       
        /// <summary>
        /// version library
        /// </summary>
        /// <returns> version </returns>
        public string versionLib()
        {
            string ver = "";
            snVersionLib(ver);

            return ver;
        }     
    }
}
