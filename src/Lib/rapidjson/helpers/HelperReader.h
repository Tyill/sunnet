#pragma once

#include "../document.h"
#include "../istreamwrapper.h"
#include "../error/en.h"
#include "HelperExtraction.h"
#include <functional>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>

namespace rapidjson
{
	namespace helper_reader
	{
		template<typename TObject, typename TSymbolPath>
		ParseErrorCode DeserializeFromFile(const TSymbolPath *pcPathToFile, TObject &tObject, size_t &uiOffsetError)
		{
			std::wifstream ifs;
			ifs.open(pcPathToFile, std::wifstream::in);
			rapidjson::WIStreamWrapper isw(ifs);
			rapidjson::GenericDocument<UTF16<wchar_t>> document;
			const auto eParseError = document.ParseStream<kParseDefaultFlags, UTF8<wchar_t>>(isw).GetParseError();
			if (eParseError == rapidjson::kParseErrorNone)
			{
				assert(document.IsObject());
				const auto& tValue = (const GenericValue<UTF8<wchar_t>>&)document;
				tObject.Deserialize(tValue);
			}
			else
				uiOffsetError = document.GetErrorOffset();
			return eParseError;
		}

		typedef GenericValue<UTF16<wchar_t>> TValueJson;

		template<typename TSymbolPath>
		ParseErrorCode DeserializeObjectsFromFile(const TSymbolPath *pcPathToFile, const std::function<void(const TValueJson&)> &funDeserialize, size_t &uiOffsetError)
		{
			std::wifstream ifs;
			ifs.open(pcPathToFile, std::wifstream::in);
			rapidjson::WIStreamWrapper isw(ifs);
			rapidjson::GenericDocument<UTF16<wchar_t>> document;
			const auto eParseError = document.ParseStream<kParseDefaultFlags, UTF8<wchar_t>>(isw).GetParseError();
			if (eParseError == rapidjson::kParseErrorNone)
			{
				assert(document.IsArray());
				for (auto itObject = document.Begin(); itObject != document.End(); ++itObject)
				{
					funDeserialize(*itObject);
				}
			}
			else
				uiOffsetError = document.GetErrorOffset();
			return eParseError;
		}

		static std::wstring GetTextFromError(const wchar_t *pcPathToFile, ParseErrorCode eParseError, size_t uiOffsetError)
		{
			std::wstringstream ss;
			const auto* pcError = rapidjson::GetParseError_En(eParseError);
			ss << L"JSON parse with error " << pcError << L" ";
			if (eParseError != kParseErrorDocumentEmpty)
				ss << L"(Offset = " << uiOffsetError << L") ";
			ss << L"file \"" << pcPathToFile << L"\"";
			return ss.str();
		}
	}
}