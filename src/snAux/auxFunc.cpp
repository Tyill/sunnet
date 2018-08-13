
#include <thread>
#include <chrono>
#include "stdafx.h"

using namespace std;

namespace SN_Aux{

	// тек дата-время %Y-%m-%d %H:%M:%S:%MS
	string CurrDateTimeMs() {

		time_t ct = time(nullptr);
		tm* lct = localtime(&ct);

		uint64_t ms = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
		uint64_t mspr = ms / 1000;
		ms -= mspr * 1000;

		char curDate[32];
		strftime(curDate, 32, "%Y-%m-%d %H:%M:%S:", lct);

		sprintf(curDate, "%s%03d", curDate, ms);

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


	bool is_number(const std::string& s)
	{
		return !s.empty() && std::find_if(s.begin(),
			s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
	}

	void sleepMs(int ms){
		std::this_thread::sleep_for(std::chrono::milliseconds(ms));
	}
}