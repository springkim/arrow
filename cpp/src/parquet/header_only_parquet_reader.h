// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_map>

// Forward declarations for Arrow types
namespace arrow {
namespace io {
class RandomAccessFile;
class ReadRange;
class IOContext;
class CacheOptions;
}  // namespace io

class Array;
class ChunkedArray;
class MemoryPool;
class ResizableBuffer;

namespace bit_util {
class BitReader;
}  // namespace bit_util

namespace util {
class RleDecoder;
}  // namespace util

template <typename T>
class Result;

template <typename T>
class Future;
}  // namespace arrow

namespace parquet {

// Basic type definitions
enum class Type {
  BOOLEAN = 0,
  INT32 = 1,
  INT64 = 2,
  INT96 = 3,
  FLOAT = 4,
  DOUBLE = 5,
  BYTE_ARRAY = 6,
  FIXED_LEN_BYTE_ARRAY = 7,
  // UNDEFINED type is not a real type and is used to represent any type not
  // covered in this enumeration.
  UNDEFINED = 8
};

enum class LogicalType {
  NONE = 0,
  UTF8 = 1,
  MAP = 2,
  MAP_KEY_VALUE = 3,
  LIST = 4,
  ENUM = 5,
  DECIMAL = 6,
  DATE = 7,
  TIME_MILLIS = 8,
  TIME_MICROS = 9,
  TIMESTAMP_MILLIS = 10,
  TIMESTAMP_MICROS = 11,
  UINT_8 = 12,
  UINT_16 = 13,
  UINT_32 = 14,
  UINT_64 = 15,
  INT_8 = 16,
  INT_16 = 17,
  INT_32 = 18,
  INT_64 = 19,
  JSON = 20,
  BSON = 21,
  INTERVAL = 22,
  UNDEFINED = 23
};

enum class Compression {
  UNCOMPRESSED = 0,
  SNAPPY = 1,
  GZIP = 2,
  LZO = 3,
  BROTLI = 4,
  LZ4 = 5,
  ZSTD = 6,
  LZ4_HADOOP = 7,
  UNDEFINED = 8
};

enum class Encoding {
  PLAIN = 0,
  PLAIN_DICTIONARY = 2,
  RLE = 3,
  BIT_PACKED = 4,
  DELTA_BINARY_PACKED = 5,
  DELTA_LENGTH_BYTE_ARRAY = 6,
  DELTA_BYTE_ARRAY = 7,
  RLE_DICTIONARY = 8,
  BYTE_STREAM_SPLIT = 9,
  UNDEFINED = 10
};

enum class ExposedEncoding {
  DICTIONARY = 0,
  UNDEFINED = 1
};

// Basic type structs
struct ByteArray {
  uint32_t len;
  const uint8_t* ptr;

  ByteArray() : len(0), ptr(nullptr) {}
  ByteArray(uint32_t len, const uint8_t* ptr) : len(len), ptr(ptr) {}
};

struct FixedLenByteArray {
  const uint8_t* ptr;

  FixedLenByteArray() : ptr(nullptr) {}
  explicit FixedLenByteArray(const uint8_t* ptr) : ptr(ptr) {}
};

struct Int96 {
  uint32_t value[3];
};

// Type traits
template <Type TYPE>
struct type_traits {};

template <>
struct type_traits<Type::BOOLEAN> {
  using c_type = bool;
};

template <>
struct type_traits<Type::INT32> {
  using c_type = int32_t;
};

template <>
struct type_traits<Type::INT64> {
  using c_type = int64_t;
};

template <>
struct type_traits<Type::INT96> {
  using c_type = Int96;
};

template <>
struct type_traits<Type::FLOAT> {
  using c_type = float;
};

template <>
struct type_traits<Type::DOUBLE> {
  using c_type = double;
};

template <>
struct type_traits<Type::BYTE_ARRAY> {
  using c_type = ByteArray;
};

template <>
struct type_traits<Type::FIXED_LEN_BYTE_ARRAY> {
  using c_type = FixedLenByteArray;
};

// Concrete type definitions
using BooleanType = type_traits<Type::BOOLEAN>;
using Int32Type = type_traits<Type::INT32>;
using Int64Type = type_traits<Type::INT64>;
using Int96Type = type_traits<Type::INT96>;
using FloatType = type_traits<Type::FLOAT>;
using DoubleType = type_traits<Type::DOUBLE>;
using ByteArrayType = type_traits<Type::BYTE_ARRAY>;
using FLBAType = type_traits<Type::FIXED_LEN_BYTE_ARRAY>;

// Forward declarations
class ColumnDescriptor;
class RowGroupMetaData;
class FileMetaData;
class ColumnReader;
class PageReader;
class Page;
class Decryptor;
class EncodedStatistics;
class ArrowInputStream;

// Exception class
class ParquetException : public std::exception {
 public:
  explicit ParquetException(const std::string& msg) : msg_(msg) {}
  ParquetException(const std::string& msg, const std::exception& e) : msg_(msg) {
    std::stringstream ss;
    ss << msg_ << ": " << e.what();
    msg_ = ss.str();
  }
  ~ParquetException() throw() {}
  const char* what() const throw() { return msg_.c_str(); }

 private:
  std::string msg_;
};

// Reader properties
class ReaderProperties {
 public:
  ReaderProperties() : use_buffered_stream_(false), buffer_size_(0) {}

  void enable_buffered_stream() { use_buffered_stream_ = true; }
  void disable_buffered_stream() { use_buffered_stream_ = false; }
  bool is_buffered_stream_enabled() const { return use_buffered_stream_; }

  void set_buffer_size(int64_t buffer_size) { buffer_size_ = buffer_size; }
  int64_t buffer_size() const { return buffer_size_; }

  void set_memory_pool(::arrow::MemoryPool* pool) { pool_ = pool; }
  ::arrow::MemoryPool* memory_pool() const { return pool_; }

 private:
  bool use_buffered_stream_;
  int64_t buffer_size_;
  ::arrow::MemoryPool* pool_ = ::arrow::default_memory_pool();
};

inline ReaderProperties default_reader_properties() {
  return ReaderProperties();
}

// Crypto context
struct CryptoContext {
  CryptoContext(bool start_with_dictionary_page, int16_t rg_ordinal, int16_t col_ordinal,
                std::shared_ptr<Decryptor> meta, std::shared_ptr<Decryptor> data)
      : start_decrypt_with_dictionary_page(start_with_dictionary_page),
        row_group_ordinal(rg_ordinal),
        column_ordinal(col_ordinal),
        meta_decryptor(std::move(meta)),
        data_decryptor(std::move(data)) {}
  CryptoContext() {}

  bool start_decrypt_with_dictionary_page = false;
  int16_t row_group_ordinal = -1;
  int16_t column_ordinal = -1;
  std::shared_ptr<Decryptor> meta_decryptor;
  std::shared_ptr<Decryptor> data_decryptor;
};

// Data page statistics
struct DataPageStats {
  DataPageStats(const EncodedStatistics* encoded_statistics, int32_t num_values,
                std::optional<int32_t> num_rows)
      : encoded_statistics(encoded_statistics),
        num_values(num_values),
        num_rows(num_rows) {}

  // Encoded statistics extracted from the page header.
  // Nullptr if there are no statistics in the page header.
  const EncodedStatistics* encoded_statistics;
  // Number of values stored in the page. Filled for both V1 and V2 data pages.
  // For repeated fields, this can be greater than number of rows. For
  // non-repeated fields, this will be the same as the number of rows.
  int32_t num_values;
  // Number of rows stored in the page. std::nullopt if not available.
  std::optional<int32_t> num_rows;
};

// Page reader interface
class PageReader {
  using DataPageFilter = std::function<bool(const DataPageStats&)>;

 public:
  virtual ~PageReader() = default;

  static std::unique_ptr<PageReader> Open(
      std::shared_ptr<ArrowInputStream> stream, int64_t total_num_values,
      Compression::type codec, bool always_compressed = false,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool(),
      const CryptoContext* ctx = nullptr);
  
  static std::unique_ptr<PageReader> Open(std::shared_ptr<ArrowInputStream> stream,
                                          int64_t total_num_values,
                                          Compression::type codec,
                                          const ReaderProperties& properties,
                                          bool always_compressed = false,
                                          const CryptoContext* ctx = nullptr);

  void set_data_page_filter(DataPageFilter data_page_filter) {
    data_page_filter_ = std::move(data_page_filter);
  }

  virtual std::shared_ptr<Page> NextPage() = 0;
  virtual void set_max_page_header_size(uint32_t size) = 0;

 protected:
  DataPageFilter data_page_filter_;
};

// Column reader interface
class ColumnReader {
 public:
  virtual ~ColumnReader() = default;

  static std::shared_ptr<ColumnReader> Make(
      const ColumnDescriptor* descr, std::unique_ptr<PageReader> pager,
      ::arrow::MemoryPool* pool = ::arrow::default_memory_pool());

  virtual bool HasNext() = 0;
  virtual Type::type type() const = 0;
  virtual const ColumnDescriptor* descr() const = 0;
  virtual ExposedEncoding GetExposedEncoding() = 0;

 protected:
  friend class RowGroupReader;
  virtual void SetExposedEncoding(ExposedEncoding encoding) = 0;
};

// Typed column reader interface
template <typename DType>
class TypedColumnReader : public ColumnReader {
 public:
  using T = typename DType::c_type;

  virtual int64_t ReadBatch(int64_t batch_size, int16_t* def_levels, int16_t* rep_levels,
                            T* values, int64_t* values_read) = 0;

  virtual int64_t Skip(int64_t num_values_to_skip) = 0;

  virtual int64_t ReadBatchWithDictionary(int64_t batch_size, int16_t* def_levels,
                                          int16_t* rep_levels, int32_t* indices,
                                          int64_t* indices_read, const T** dict,
                                          int32_t* dict_len) = 0;
};

// Specialized column reader types
using BoolReader = TypedColumnReader<BooleanType>;
using Int32Reader = TypedColumnReader<Int32Type>;
using Int64Reader = TypedColumnReader<Int64Type>;
using Int96Reader = TypedColumnReader<Int96Type>;
using FloatReader = TypedColumnReader<FloatType>;
using DoubleReader = TypedColumnReader<DoubleType>;
using ByteArrayReader = TypedColumnReader<ByteArrayType>;
using FixedLenByteArrayReader = TypedColumnReader<FLBAType>;

// Row group reader
class RowGroupReader {
 public:
  struct Contents {
    virtual ~Contents() {}
    virtual std::unique_ptr<PageReader> GetColumnPageReader(int i) = 0;
    virtual const RowGroupMetaData* metadata() const = 0;
    virtual const ReaderProperties* properties() const = 0;
  };

  explicit RowGroupReader(std::unique_ptr<Contents> contents) : contents_(std::move(contents)) {}

  const RowGroupMetaData* metadata() const { return contents_->metadata(); }

  std::shared_ptr<ColumnReader> Column(int i) {
    std::unique_ptr<PageReader> pager = GetColumnPageReader(i);
    const ColumnDescriptor* descr = metadata()->schema()->Column(i);
    return ColumnReader::Make(descr, std::move(pager), contents_->properties()->memory_pool());
  }

  std::shared_ptr<ColumnReader> ColumnWithExposeEncoding(int i, ExposedEncoding encoding_to_expose) {
    auto reader = Column(i);
    reader->SetExposedEncoding(encoding_to_expose);
    return reader;
  }

  std::unique_ptr<PageReader> GetColumnPageReader(int i) {
    return contents_->GetColumnPageReader(i);
  }

 private:
  std::unique_ptr<Contents> contents_;
};

// Parquet file reader
class ParquetFileReader {
 public:
  struct Contents {
    static std::unique_ptr<Contents> Open(
        std::shared_ptr<::arrow::io::RandomAccessFile> source,
        const ReaderProperties& props = default_reader_properties(),
        std::shared_ptr<FileMetaData> metadata = nullptr);

    static ::arrow::Future<std::unique_ptr<Contents>> OpenAsync(
        std::shared_ptr<::arrow::io::RandomAccessFile> source,
        const ReaderProperties& props = default_reader_properties(),
        std::shared_ptr<FileMetaData> metadata = nullptr);

    virtual ~Contents() = default;
    virtual void Close() = 0;
    virtual std::shared_ptr<RowGroupReader> GetRowGroup(int i) = 0;
    virtual std::shared_ptr<FileMetaData> metadata() const = 0;
  };

  ParquetFileReader() {}
  ~ParquetFileReader() {}

  static std::unique_ptr<ParquetFileReader> Open(
      std::shared_ptr<::arrow::io::RandomAccessFile> source,
      const ReaderProperties& props = default_reader_properties(),
      std::shared_ptr<FileMetaData> metadata = nullptr) {
    auto contents = Contents::Open(source, props, metadata);
    auto reader = std::unique_ptr<ParquetFileReader>(new ParquetFileReader());
    reader->Open(std::move(contents));
    return reader;
  }

  static std::unique_ptr<ParquetFileReader> OpenFile(
      const std::string& path, bool memory_map = false,
      const ReaderProperties& props = default_reader_properties(),
      std::shared_ptr<FileMetaData> metadata = nullptr) {
    // Implementation would create a RandomAccessFile from the path
    // and call Open() with it
    throw ParquetException("OpenFile not implemented in header-only version");
  }

  static ::arrow::Future<std::unique_ptr<ParquetFileReader>> OpenAsync(
      std::shared_ptr<::arrow::io::RandomAccessFile> source,
      const ReaderProperties& props = default_reader_properties(),
      std::shared_ptr<FileMetaData> metadata = nullptr) {
    // Implementation would asynchronously open the file
    throw ParquetException("OpenAsync not implemented in header-only version");
  }

  void Open(std::unique_ptr<Contents> contents) {
    contents_ = std::move(contents);
  }

  void Close() {
    if (contents_) {
      contents_->Close();
    }
  }

  std::shared_ptr<RowGroupReader> RowGroup(int i) {
    return contents_->GetRowGroup(i);
  }

  std::shared_ptr<FileMetaData> metadata() const {
    return contents_->metadata();
  }

 private:
  std::unique_ptr<Contents> contents_;
};

// Helper function to scan file contents
inline int64_t ScanFileContents(std::vector<int> columns, const int32_t column_batch_size,
                         ParquetFileReader* reader) {
  int64_t rows_read = 0;
  for (int r = 0; r < reader->metadata()->num_row_groups(); ++r) {
    auto group = reader->RowGroup(r);
    int64_t batch_size = column_batch_size;
    std::vector<int64_t> values_read(columns.size());
    std::vector<int64_t> rows_read_per_batch(columns.size());

    for (size_t i = 0; i < columns.size(); ++i) {
      const int col = columns[i];
      auto column_reader = group->Column(col);
      
      switch (column_reader->type()) {
        case Type::BOOLEAN: {
          auto reader = std::static_pointer_cast<BoolReader>(column_reader);
          std::vector<bool> values(batch_size);
          while (reader->HasNext()) {
            reader->ReadBatch(batch_size, nullptr, nullptr, values.data(), &values_read[i]);
            rows_read_per_batch[i] += values_read[i];
          }
          break;
        }
        case Type::INT32: {
          auto reader = std::static_pointer_cast<Int32Reader>(column_reader);
          std::vector<int32_t> values(batch_size);
          while (reader->HasNext()) {
            reader->ReadBatch(batch_size, nullptr, nullptr, values.data(), &values_read[i]);
            rows_read_per_batch[i] += values_read[i];
          }
          break;
        }
        case Type::INT64: {
          auto reader = std::static_pointer_cast<Int64Reader>(column_reader);
          std::vector<int64_t> values(batch_size);
          while (reader->HasNext()) {
            reader->ReadBatch(batch_size, nullptr, nullptr, values.data(), &values_read[i]);
            rows_read_per_batch[i] += values_read[i];
          }
          break;
        }
        case Type::FLOAT: {
          auto reader = std::static_pointer_cast<FloatReader>(column_reader);
          std::vector<float> values(batch_size);
          while (reader->HasNext()) {
            reader->ReadBatch(batch_size, nullptr, nullptr, values.data(), &values_read[i]);
            rows_read_per_batch[i] += values_read[i];
          }
          break;
        }
        case Type::DOUBLE: {
          auto reader = std::static_pointer_cast<DoubleReader>(column_reader);
          std::vector<double> values(batch_size);
          while (reader->HasNext()) {
            reader->ReadBatch(batch_size, nullptr, nullptr, values.data(), &values_read[i]);
            rows_read_per_batch[i] += values_read[i];
          }
          break;
        }
        case Type::BYTE_ARRAY: {
          auto reader = std::static_pointer_cast<ByteArrayReader>(column_reader);
          std::vector<ByteArray> values(batch_size);
          while (reader->HasNext()) {
            reader->ReadBatch(batch_size, nullptr, nullptr, values.data(), &values_read[i]);
            rows_read_per_batch[i] += values_read[i];
          }
          break;
        }
        case Type::FIXED_LEN_BYTE_ARRAY: {
          auto reader = std::static_pointer_cast<FixedLenByteArrayReader>(column_reader);
          std::vector<FixedLenByteArray> values(batch_size);
          while (reader->HasNext()) {
            reader->ReadBatch(batch_size, nullptr, nullptr, values.data(), &values_read[i]);
            rows_read_per_batch[i] += values_read[i];
          }
          break;
        }
        default:
          throw ParquetException("Not implemented type");
      }
    }

    if (!columns.empty()) {
      rows_read += rows_read_per_batch[0];
    }
  }
  return rows_read;
}

// Simple usage example
inline void ReadParquetFile(const std::string& filename) {
  try {
    // Open the file
    auto reader = ParquetFileReader::OpenFile(filename);
    
    // Get the file metadata
    auto file_metadata = reader->metadata();
    int num_row_groups = file_metadata->num_row_groups();
    
    std::cout << "Parquet file has " << num_row_groups << " row groups." << std::endl;
    
    // Read all row groups
    for (int i = 0; i < num_row_groups; i++) {
      auto row_group = reader->RowGroup(i);
      int num_columns = row_group->metadata()->num_columns();
      
      std::cout << "Row group " << i << " has " << num_columns << " columns." << std::endl;
      
      // Read all columns in the row group
      for (int j = 0; j < num_columns; j++) {
        auto column_reader = row_group->Column(j);
        
        std::cout << "Column " << j << " has type: ";
        switch (column_reader->type()) {
          case Type::BOOLEAN:
            std::cout << "BOOLEAN" << std::endl;
            break;
          case Type::INT32:
            std::cout << "INT32" << std::endl;
            break;
          case Type::INT64:
            std::cout << "INT64" << std::endl;
            break;
          case Type::FLOAT:
            std::cout << "FLOAT" << std::endl;
            break;
          case Type::DOUBLE:
            std::cout << "DOUBLE" << std::endl;
            break;
          case Type::BYTE_ARRAY:
            std::cout << "BYTE_ARRAY" << std::endl;
            break;
          case Type::FIXED_LEN_BYTE_ARRAY:
            std::cout << "FIXED_LEN_BYTE_ARRAY" << std::endl;
            break;
          default:
            std::cout << "UNKNOWN" << std::endl;
            break;
        }
      }
    }
    
    // Close the reader
    reader->Close();
  } catch (const ParquetException& e) {
    std::cerr << "Error reading Parquet file: " << e.what() << std::endl;
  }
}

} // namespace parquet
