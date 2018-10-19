using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SN_API;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            Net net = new Net();

            string ver = net.versionLib();
        }
    }              
            
}
