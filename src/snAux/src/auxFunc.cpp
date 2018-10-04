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
#include <thread>
#include <chrono>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include "stdafx.h"

#ifdef WIN32
#include <windows.h>
#endif

using namespace std;

namespace SN_Aux{

    // %Y-%m-%d %H:%M:%S:%MS
    string CurrDateTimeMs() {

        time_t ct = time(nullptr);
        tm* lct = localtime(&ct);

        uint64_t ms = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
        uint64_t mspr = ms / 1000;
        ms -= mspr * 1000;

        char curDate[32];
        strftime(curDate, 32, "%Y-%m-%d %H:%M:%S:", lct);

        sprintf(curDate, "%s%03d", curDate, int(ms));

        return curDate;
    }

    vector<string> split(string text, const char* sep)
    {
        char* cstr = (char*)text.c_str();

        vector<string> res;
        char* pch = strtok(cstr, sep);
        while (pch != NULL){
            res.push_back(string(pch));
            pch = strtok(NULL, sep);
        }

        return res;
    }

    std::string trim(const std::string& str)
    {
        size_t first = str.find_first_not_of(' ');
        if (std::string::npos == first){
            return str;
        }
        size_t last = str.find_last_not_of(' ');
        return str.substr(first, (last - first + 1));
    }

    std::string toLower(const std::string& str)
    {
        string out = str;
        std::transform(str.begin(), str.end(), out.begin(), ::tolower);

        return out;
    }

    bool is_number(const std::string& s)
    {
        return !s.empty() && std::find_if(s.begin(),
            s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
    }

    void sleepMs(int ms){
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }

    bool createSubDirectory(const std::string& strDirs){

        size_t sz = strDirs.size(); int ret = 0;
        string strTmp = "";
        for (size_t i = 0; i < sz; ++i) {
            char ch = strDirs[i];
            if (ch != '\\' && ch != '/') strTmp += ch;
            else {
#if defined(_WIN32)
                strTmp += "\\";
                ret = CreateDirectoryA(strTmp.c_str(), NULL);
#else
                strTmp += "/";
                ret = mkdir(strTmp.c_str(), 0733);
#endif
            }
        }
        return ret == 0;
    }
}