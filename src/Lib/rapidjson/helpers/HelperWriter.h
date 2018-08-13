#pragma once

#include "../prettywriter.h"
#include "../ostreamwrapper.h"
#include <fstream>
#include <functional>

namespace rapidjson
{
	namespace helper_writer
	{
		typedef rapidjson::PrettyWriter<rapidjson::WOStreamWrapper, UTF16<wchar_t>, UTF8<wchar_t>, CrtAllocator, kWriteNanAndInfFlag> TWriterJson;

		template<typename TSymbolPath, typename TObject>
		void SerializeToFile(const TSymbolPath *pcPathToFile, const TObject &tObject)
		{
			std::wofstream ofs;
			ofs.open(pcPathToFile, std::wofstream::out | std::wofstream::trunc);
			rapidjson::WOStreamWrapper osw(ofs);
			TWriterJson writer(osw);
			tObject.Serialize(writer);
			ofs.close();
		}

		template<typename TObject, typename TSymbolPath, typename InputIterator>
		void SerializeToFile(const TSymbolPath *pcPathToFile, InputIterator itFirst, InputIterator itLast, const std::function<void(TWriterJson&, const TObject&)> &funSerialize)
		{
			std::wofstream ofs;
			ofs.open(pcPathToFile, std::wofstream::out | std::wofstream::trunc);
			rapidjson::WOStreamWrapper osw(ofs);
			TWriterJson writer(osw);
			writer.StartArray();
			for (auto it = itFirst; it != itLast; ++it)
			{
				funSerialize(writer, *it);
			}
			writer.EndArray();
			ofs.close();
		}
	}
}