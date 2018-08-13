
#pragma once

#include <vector>
#include <string>

namespace SN_Aux{
	
	// тек дата-время %Y-%m-%d %H:%M:%S:%MS
	std::string CurrDateTimeMs();

	std::vector<std::string> split(std::string text, const char* sep);

	std::string trim(const std::string& str);

	void sleepMs(int ms);

	bool is_number(const std::string& s);
}