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

#include <gtest/gtest.h>

#include <stdio.h>

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
#include <arrow/array.h>
#include <arrow/c/bridge.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/api.h>
#include <arrow/json/from_string.h>
#endif

#include "nanoarrow/nanoarrow_ipc.hpp"

TEST(NanoarrowIpcWriter, OutputStreamBuffer) {
  struct ArrowError error;

  // The output buffer starts with some header
  std::string header = "HELLO WORLD";
  nanoarrow::UniqueBuffer output;
  ASSERT_EQ(ArrowBufferAppend(output.get(), header.data(), header.size()), NANOARROW_OK);

  // Then the stream starts appending to it
  nanoarrow::ipc::UniqueOutputStream stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream.get(), output.get()), NANOARROW_OK);

  std::string message = "\n-_-_";
  for (int i = 0; i < 4; ++i) {
    int64_t actually_written;
    ASSERT_EQ(stream->write(stream.get(), message.data(), message.size(),
                            &actually_written, &error),
              NANOARROW_OK)
        << error.message;
    EXPECT_EQ(actually_written, message.size());
  }

  EXPECT_EQ(output->size_bytes, header.size() + 4 * message.size());

  std::vector<char> output_str(output->size_bytes, '\0');
  memcpy(output_str.data(), output->data, output->size_bytes);
  EXPECT_EQ(std::string(output_str.data(), output_str.size()),
            header + message + message + message + message);
}

// clang-tidy helpfully reminds us that file_ptr might not be released
// if an assertion fails
struct FileCloser {
  FileCloser(FILE* file) : file_(file) {}
  ~FileCloser() {
    if (file_) fclose(file_);
  }
  FILE* file_{};
};

TEST(NanoarrowIpcWriter, OutputStreamFile) {
  FILE* file_ptr = tmpfile();
  FileCloser closer{file_ptr};
  ASSERT_NE(file_ptr, nullptr);

  // Start by writing some header
  std::string header = "HELLO WORLD";
  ASSERT_EQ(fwrite(header.data(), 1, header.size(), file_ptr), header.size());

  // Then seek to test that we overwrite WORLD but not HELLO
  fseek(file_ptr, 6, SEEK_SET);

  nanoarrow::ipc::UniqueOutputStream stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitFile(stream.get(), file_ptr, /*close_on_release=*/1),
            NANOARROW_OK);
  closer.file_ = nullptr;

  struct ArrowError error;

  // Start appending using the stream
  std::string message = "\n-_-_";
  for (int i = 0; i < 4; ++i) {
    int64_t actually_written;
    ASSERT_EQ(stream->write(stream.get(), message.data(), message.size(),
                            &actually_written, &error),
              NANOARROW_OK)
        << error.message;
    EXPECT_EQ(actually_written, message.size());
  }

  // Read back the whole file
  fseek(file_ptr, 0, SEEK_END);
  std::vector<char> buffer(static_cast<size_t>(ftell(file_ptr)), '\0');
  rewind(file_ptr);
  ASSERT_EQ(fread(buffer.data(), 1, buffer.size(), file_ptr), buffer.size());

  EXPECT_EQ(buffer.size(), 6 + 4 * message.size());
  EXPECT_EQ(std::string(buffer.data(), buffer.size()),
            "HELLO " + message + message + message + message);
}

TEST(NanoarrowIpcWriter, OutputStreamFileError) {
  nanoarrow::ipc::UniqueOutputStream stream;
  errno = EINVAL;
  EXPECT_EQ(ArrowIpcOutputStreamInitFile(stream.get(), nullptr, /*close_on_release=*/1),
            EINVAL);

  auto phony_path = __FILE__ + std::string(".phony");
  FILE* file_ptr = fopen(phony_path.c_str(), "rb");
  FileCloser closer{file_ptr};
  ASSERT_EQ(file_ptr, nullptr);
  EXPECT_EQ(ArrowIpcOutputStreamInitFile(stream.get(), file_ptr, /*close_on_release=*/1),
            ENOENT);
  closer.file_ = nullptr;
}

struct ArrowIpcWriterPrivate {
  struct ArrowIpcEncoder encoder;
  struct ArrowIpcOutputStream output_stream;
  struct ArrowBuffer buffer;
  struct ArrowBuffer body_buffer;

  int writing_file;
  int64_t bytes_written;
  struct ArrowIpcFooter footer;
};

#define NANOARROW_IPC_FILE_PADDED_MAGIC "ARROW1\0"

TEST(NanoarrowIpcWriter, FileWriting) {
  struct ArrowError error;

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream.get(), output.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), stream.get()), NANOARROW_OK);

  // the writer starts out in stream mode
  auto* p = static_cast<struct ArrowIpcWriterPrivate*>(writer->private_data);
  EXPECT_FALSE(p->writing_file);
  EXPECT_EQ(p->bytes_written, 0);
  EXPECT_EQ(p->footer.schema.release, nullptr);
  EXPECT_EQ(p->footer.record_batch_blocks.size_bytes, 0);

  // now it switches to file mode
  EXPECT_EQ(ArrowIpcWriterStartFile(writer.get(), &error), NANOARROW_OK) << error.message;
  EXPECT_TRUE(p->writing_file);
  // and has written the leading magic
  EXPECT_EQ(p->bytes_written, sizeof(NANOARROW_IPC_FILE_PADDED_MAGIC));
  // but not a schema or any record batches
  EXPECT_EQ(p->footer.schema.release, nullptr);
  EXPECT_EQ(p->footer.record_batch_blocks.size_bytes, 0);

  // write a schema
  nanoarrow::UniqueSchema schema;
  ASSERT_EQ(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  EXPECT_EQ(ArrowIpcWriterWriteSchema(writer.get(), schema.get(), &error), NANOARROW_OK)
      << error.message;
  // more has been written
  auto after_schema = p->bytes_written;
  EXPECT_GT(after_schema, sizeof(NANOARROW_IPC_FILE_PADDED_MAGIC));
  // the schema is cached in the writer's footer for later finalization
  EXPECT_NE(p->footer.schema.release, nullptr);
  // still no record batches
  EXPECT_EQ(p->footer.record_batch_blocks.size_bytes, 0);

  // write a batch
  nanoarrow::UniqueArray array;
  nanoarrow::UniqueArrayView array_view;
  ASSERT_EQ(ArrowArrayInitFromSchema(array.get(), schema.get(), &error), NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(array_view.get(), schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(array_view.get(), array.get(), &error), NANOARROW_OK)
      << error.message;
  EXPECT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), array_view.get(), &error),
            NANOARROW_OK)
      << error.message;
  // more has been written
  auto after_batch = p->bytes_written;
  EXPECT_GT(after_batch, after_schema);
  // one record batch's block is stored
  EXPECT_EQ(p->footer.record_batch_blocks.size_bytes, sizeof(struct ArrowIpcFileBlock));

  // end the stream
  EXPECT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), nullptr, &error), NANOARROW_OK)
      << error.message;
  // more has been written
  auto after_eos = p->bytes_written;
  EXPECT_GT(after_eos, after_batch);
  // EOS isn't stored in the blocks
  EXPECT_EQ(p->footer.record_batch_blocks.size_bytes, sizeof(struct ArrowIpcFileBlock));

  // finalize the file
  EXPECT_EQ(ArrowIpcWriterFinalizeFile(writer.get(), &error), NANOARROW_OK)
      << error.message;
  // more has been written
  auto after_footer = p->bytes_written;
  EXPECT_GT(after_footer, after_eos);
}

TEST(NanoarrowIpcWriter, WriteDictionaryBatch) {
  struct ArrowError error;

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream.get(), output.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), stream.get()), NANOARROW_OK);

  auto* p = static_cast<struct ArrowIpcWriterPrivate*>(writer->private_data);

  // Build a simple Utf8 values array
  nanoarrow::UniqueSchema values_schema;
  ASSERT_EQ(ArrowSchemaInitFromType(values_schema.get(), NANOARROW_TYPE_STRING),
            NANOARROW_OK);

  nanoarrow::UniqueArray values_array;
  ASSERT_EQ(ArrowArrayInitFromSchema(values_array.get(), values_schema.get(), nullptr),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(values_array.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(values_array.get(), ArrowCharView("foo")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(values_array.get(), ArrowCharView("bar")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(values_array.get(), &error), NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueArrayView values_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(values_view.get(), values_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(values_view.get(), values_array.get(), &error),
            NANOARROW_OK)
      << error.message;

  // stream mode: write a DictionaryBatch — bytes are emitted but no block is tracked
  EXPECT_EQ(p->bytes_written, 0);
  EXPECT_EQ(p->footer.dictionary_blocks.size_bytes, 0);

  EXPECT_EQ(ArrowIpcWriterWriteDictionaryBatch(writer.get(), /*dictionary_id=*/0,
                                               /*is_delta=*/0, values_view.get(), &error),
            NANOARROW_OK)
      << error.message;

  auto after_dict_stream = p->bytes_written;
  EXPECT_GT(after_dict_stream, 0);
  // no block tracked in stream mode
  EXPECT_EQ(p->footer.dictionary_blocks.size_bytes, 0);

  // file mode: the block is tracked in the footer
  nanoarrow::ipc::UniqueOutputStream stream2;
  nanoarrow::UniqueBuffer output2;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(stream2.get(), output2.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer2;
  ASSERT_EQ(ArrowIpcWriterInit(writer2.get(), stream2.get()), NANOARROW_OK);

  auto* p2 = static_cast<struct ArrowIpcWriterPrivate*>(writer2->private_data);

  ASSERT_EQ(ArrowIpcWriterStartFile(writer2.get(), &error), NANOARROW_OK)
      << error.message;
  EXPECT_EQ(p2->footer.dictionary_blocks.size_bytes, 0);

  EXPECT_EQ(ArrowIpcWriterWriteDictionaryBatch(writer2.get(), /*dictionary_id=*/0,
                                               /*is_delta=*/0, values_view.get(), &error),
            NANOARROW_OK)
      << error.message;

  // one block tracked in file mode
  EXPECT_EQ(p2->footer.dictionary_blocks.size_bytes, sizeof(struct ArrowIpcFileBlock));
}

TEST(NanoarrowIpcWriter, RoundtripDeltaDictionaryStream) {
  struct ArrowError error;
  nanoarrow::UniqueSchema schema;
  ASSERT_EQ(ArrowSchemaInitFromType(schema.get(), NANOARROW_TYPE_STRUCT),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(schema.get(), 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(schema->children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema->children[0], "dict_col"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateDictionary(schema->children[0]), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(schema->children[0]->dictionary,
                                    NANOARROW_TYPE_STRING),
            NANOARROW_OK);

  nanoarrow::UniqueSchema values_schema;
  ASSERT_EQ(ArrowSchemaInitFromType(values_schema.get(), NANOARROW_TYPE_STRING),
            NANOARROW_OK);
  nanoarrow::UniqueArray full_values;
  nanoarrow::UniqueArray delta_values;
  ASSERT_EQ(ArrowArrayInitFromSchema(full_values.get(), values_schema.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromSchema(delta_values.get(), values_schema.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(full_values.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(delta_values.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(full_values.get(), ArrowCharView("zero")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(delta_values.get(), ArrowCharView("one")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(delta_values.get(), ArrowCharView("two")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(full_values.get(), &error), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(delta_values.get(), &error), NANOARROW_OK);

  nanoarrow::UniqueArray batch1;
  nanoarrow::UniqueArray batch2;
  ASSERT_EQ(ArrowArrayInitFromSchema(batch1.get(), schema.get(), &error), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayInitFromSchema(batch2.get(), schema.get(), &error), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(batch1.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(batch2.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(batch1->children[0]->dictionary,
                                  ArrowCharView("zero")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(batch1->children[0], 0), NANOARROW_OK);
  batch1->length = 1;
  ASSERT_EQ(ArrowArrayAppendString(batch2->children[0]->dictionary,
                                  ArrowCharView("zero")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(batch2->children[0]->dictionary,
                                  ArrowCharView("one")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(batch2->children[0]->dictionary,
                                  ArrowCharView("two")),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(batch2->children[0], 2), NANOARROW_OK);
  batch2->length = 1;
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(batch1.get(), &error), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(batch2.get(), &error), NANOARROW_OK);

  nanoarrow::UniqueArrayView full_values_view;
  nanoarrow::UniqueArrayView delta_values_view;
  nanoarrow::UniqueArrayView batch1_view;
  nanoarrow::UniqueArrayView batch2_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(full_values_view.get(), values_schema.get(),
                                        &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewInitFromSchema(delta_values_view.get(), values_schema.get(),
                                        &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewInitFromSchema(batch1_view.get(), schema.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewInitFromSchema(batch2_view.get(), schema.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(full_values_view.get(), full_values.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(delta_values_view.get(), delta_values.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(batch1_view.get(), batch1.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(batch2_view.get(), batch2.get(), &error),
            NANOARROW_OK);

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteSchema(writer.get(), schema.get(), &error), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteDictionaryBatch(
                writer.get(), 0, /*is_delta=*/0, full_values_view.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), batch1_view.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteDictionaryBatch(
                writer.get(), 0, /*is_delta=*/1, delta_values_view.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), batch2_view.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayView(writer.get(), nullptr, &error), NANOARROW_OK);

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
  auto arrow_input = std::make_shared<arrow::io::BufferReader>(
      arrow::Buffer::Wrap(output->data, output->size_bytes));
  auto maybe_arrow_reader = arrow::ipc::RecordBatchStreamReader::Open(arrow_input);
  ASSERT_TRUE(maybe_arrow_reader.ok()) << maybe_arrow_reader.status();
  auto arrow_reader = maybe_arrow_reader.ValueUnsafe();
  std::shared_ptr<arrow::RecordBatch> arrow_batch1;
  std::shared_ptr<arrow::RecordBatch> arrow_batch2;
  ASSERT_TRUE(arrow_reader->ReadNext(&arrow_batch1).ok());
  ASSERT_TRUE(arrow_reader->ReadNext(&arrow_batch2).ok());
  auto arrow_dictionary1 =
      std::static_pointer_cast<arrow::DictionaryArray>(arrow_batch1->column(0));
  auto arrow_dictionary2 =
      std::static_pointer_cast<arrow::DictionaryArray>(arrow_batch2->column(0));
  EXPECT_EQ(arrow_dictionary1->dictionary()->length(), 1);
  EXPECT_EQ(arrow_dictionary2->dictionary()->length(), 3);
#endif

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, output.get()), NANOARROW_OK);
  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(reader.get(), &input, nullptr), NANOARROW_OK);
  nanoarrow::UniqueSchema roundtrip_schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(reader.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  nanoarrow::UniqueArray roundtrip_batch1;
  nanoarrow::UniqueArray roundtrip_batch2;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_batch1.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_batch2.get(), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(roundtrip_batch1->children[0]->dictionary->length, 1);
  EXPECT_EQ(roundtrip_batch2->children[0]->dictionary->length, 3);

  nanoarrow::UniqueArrayView roundtrip_view2;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(roundtrip_view2.get(), roundtrip_schema.get(),
                                        &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(roundtrip_view2.get(), roundtrip_batch2.get(), &error),
            NANOARROW_OK);
  EXPECT_EQ(ArrowArrayViewGetStringUnsafe(roundtrip_view2->children[0]->dictionary, 2),
            ArrowCharView("two"));

  // Files permit deltas (applied in footer order), but not replacement batches.
  nanoarrow::UniqueBuffer file_output;
  nanoarrow::ipc::UniqueOutputStream file_out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(file_out_stream.get(), file_output.get()),
            NANOARROW_OK);
  nanoarrow::ipc::UniqueWriter file_writer;
  ASSERT_EQ(ArrowIpcWriterInit(file_writer.get(), file_out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterStartFile(file_writer.get(), &error), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteSchema(file_writer.get(), schema.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteDictionaryBatch(
                file_writer.get(), 0, /*is_delta=*/0, full_values_view.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayView(file_writer.get(), batch1_view.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteDictionaryBatch(
                file_writer.get(), 0, /*is_delta=*/1, delta_values_view.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayView(file_writer.get(), batch2_view.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayView(file_writer.get(), nullptr, &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterFinalizeFile(file_writer.get(), &error), NANOARROW_OK)
      << error.message;

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
  auto arrow_file_input = std::make_shared<arrow::io::BufferReader>(
      arrow::Buffer::Wrap(file_output->data, file_output->size_bytes));
  auto maybe_file_reader = arrow::ipc::RecordBatchFileReader::Open(arrow_file_input);
  ASSERT_TRUE(maybe_file_reader.ok()) << maybe_file_reader.status();
  auto arrow_file_reader = maybe_file_reader.ValueUnsafe();
  ASSERT_EQ(arrow_file_reader->num_record_batches(), 2);
  auto maybe_file_batch2 = arrow_file_reader->ReadRecordBatch(1);
  ASSERT_TRUE(maybe_file_batch2.ok()) << maybe_file_batch2.status();
  auto arrow_file_dictionary2 = std::static_pointer_cast<arrow::DictionaryArray>(
      maybe_file_batch2.ValueUnsafe()->column(0));
  EXPECT_EQ(arrow_file_dictionary2->dictionary()->length(), 3);
#endif
}

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
struct DeltaDictionaryTypeCase {
  std::shared_ptr<arrow::DataType> value_type;
  const char* initial_values;
  const char* extended_values;
};

class DeltaDictionaryTypeTest
    : public ::testing::TestWithParam<DeltaDictionaryTypeCase> {};

static void AssertReadsArrowCppDeltaStream(
    const std::shared_ptr<arrow::Array>& dictionary_array1,
    const std::shared_ptr<arrow::Array>& dictionary_array2) {
  auto dictionary_type = dictionary_array1->type();
  auto schema = arrow::schema({arrow::field("dictionary", dictionary_type)});
  auto expected1 =
      arrow::RecordBatch::Make(schema, 4, {dictionary_array1});
  auto expected2 =
      arrow::RecordBatch::Make(schema, 4, {dictionary_array2});

  auto maybe_sink = arrow::io::BufferOutputStream::Create();
  ASSERT_TRUE(maybe_sink.ok()) << maybe_sink.status();
  auto sink = maybe_sink.ValueUnsafe();
  auto options = arrow::ipc::IpcWriteOptions::Defaults();
  options.emit_dictionary_deltas = true;
  auto maybe_writer = arrow::ipc::MakeStreamWriter(sink, schema, options);
  ASSERT_TRUE(maybe_writer.ok()) << maybe_writer.status();
  auto arrow_writer = maybe_writer.ValueUnsafe();
  ASSERT_TRUE(arrow_writer->WriteRecordBatch(*expected1).ok());
  ASSERT_TRUE(arrow_writer->WriteRecordBatch(*expected2).ok());
  ASSERT_TRUE(arrow_writer->Close().ok());
  auto maybe_buffer = sink->Finish();
  ASSERT_TRUE(maybe_buffer.ok()) << maybe_buffer.status();
  auto buffer = maybe_buffer.ValueUnsafe();

  nanoarrow::UniqueBuffer ipc_buffer;
  ASSERT_EQ(ArrowBufferAppend(ipc_buffer.get(), buffer->data(), buffer->size()),
            NANOARROW_OK);
  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, ipc_buffer.get()), NANOARROW_OK);
  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(reader.get(), &input, nullptr), NANOARROW_OK);

  struct ArrowError error;
  nanoarrow::UniqueSchema roundtrip_schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(reader.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  auto maybe_arrow_schema = arrow::ImportSchema(roundtrip_schema.get());
  ASSERT_TRUE(maybe_arrow_schema.ok()) << maybe_arrow_schema.status();
  auto arrow_schema = maybe_arrow_schema.ValueUnsafe();

  nanoarrow::UniqueArray roundtrip1;
  nanoarrow::UniqueArray roundtrip2;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip1.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip2.get(), &error),
            NANOARROW_OK)
      << error.message;
  auto maybe_roundtrip1 = arrow::ImportRecordBatch(roundtrip1.get(), arrow_schema);
  auto maybe_roundtrip2 = arrow::ImportRecordBatch(roundtrip2.get(), arrow_schema);
  ASSERT_TRUE(maybe_roundtrip1.ok()) << maybe_roundtrip1.status();
  ASSERT_TRUE(maybe_roundtrip2.ok()) << maybe_roundtrip2.status();
  EXPECT_TRUE(maybe_roundtrip1.ValueUnsafe()->Equals(*expected1));
  EXPECT_TRUE(maybe_roundtrip2.ValueUnsafe()->Equals(*expected2));
}

TEST_P(DeltaDictionaryTypeTest, ReadsArrowCppDeltaStream) {
  const DeltaDictionaryTypeCase& param = GetParam();
  auto dictionary_type = arrow::dictionary(arrow::int32(), param.value_type);
  auto maybe_array1 = arrow::json::DictArrayFromJSONString(
      dictionary_type, "[0, 1, null, 0]", param.initial_values);
  ASSERT_TRUE(maybe_array1.ok()) << maybe_array1.status();
  auto maybe_array2 = arrow::json::DictArrayFromJSONString(
      dictionary_type, "[2, 3, 1, null]", param.extended_values);
  ASSERT_TRUE(maybe_array2.ok()) << maybe_array2.status();
  AssertReadsArrowCppDeltaStream(maybe_array1.ValueUnsafe(),
                                 maybe_array2.ValueUnsafe());
}

TEST(NanoarrowIpcWriter, ReadsArrowCppDenseUnionDictionaryDelta) {
  auto type_ids1 = arrow::json::ArrayFromJSONString(arrow::int8(), "[5, 7]");
  auto type_ids2 = arrow::json::ArrayFromJSONString(arrow::int8(), "[5, 7, 5, 7]");
  auto offsets1 = arrow::json::ArrayFromJSONString(arrow::int32(), "[0, 0]");
  auto offsets2 = arrow::json::ArrayFromJSONString(arrow::int32(), "[0, 0, 1, 1]");
  auto ints1 = arrow::json::ArrayFromJSONString(arrow::int32(), "[1]");
  auto ints2 = arrow::json::ArrayFromJSONString(arrow::int32(), "[1, 2]");
  auto strings1 = arrow::json::ArrayFromJSONString(arrow::utf8(), "[\"a\"]");
  auto strings2 = arrow::json::ArrayFromJSONString(arrow::utf8(), "[\"a\", \"b\"]");
  ASSERT_TRUE(type_ids1.ok() && type_ids2.ok() && offsets1.ok() && offsets2.ok() &&
              ints1.ok() && ints2.ok() && strings1.ok() && strings2.ok());

  auto maybe_values1 = arrow::DenseUnionArray::Make(
      *type_ids1.ValueUnsafe(), *offsets1.ValueUnsafe(),
      {ints1.ValueUnsafe(), strings1.ValueUnsafe()}, {"i", "s"}, {5, 7});
  auto maybe_values2 = arrow::DenseUnionArray::Make(
      *type_ids2.ValueUnsafe(), *offsets2.ValueUnsafe(),
      {ints2.ValueUnsafe(), strings2.ValueUnsafe()}, {"i", "s"}, {5, 7});
  ASSERT_TRUE(maybe_values1.ok()) << maybe_values1.status();
  ASSERT_TRUE(maybe_values2.ok()) << maybe_values2.status();

  auto indices1 =
      arrow::json::ArrayFromJSONString(arrow::int32(), "[0, 1, null, 0]");
  auto indices2 =
      arrow::json::ArrayFromJSONString(arrow::int32(), "[2, 3, 1, null]");
  ASSERT_TRUE(indices1.ok() && indices2.ok());
  auto dictionary_type =
      arrow::dictionary(arrow::int32(), maybe_values1.ValueUnsafe()->type());
  auto maybe_array1 = arrow::DictionaryArray::FromArrays(
      dictionary_type, indices1.ValueUnsafe(), maybe_values1.ValueUnsafe());
  auto maybe_array2 = arrow::DictionaryArray::FromArrays(
      dictionary_type, indices2.ValueUnsafe(), maybe_values2.ValueUnsafe());
  ASSERT_TRUE(maybe_array1.ok()) << maybe_array1.status();
  ASSERT_TRUE(maybe_array2.ok()) << maybe_array2.status();
  AssertReadsArrowCppDeltaStream(maybe_array1.ValueUnsafe(),
                                 maybe_array2.ValueUnsafe());
}

INSTANTIATE_TEST_SUITE_P(
    NanoarrowIpcWriter, DeltaDictionaryTypeTest,
    ::testing::Values(
        DeltaDictionaryTypeCase{arrow::boolean(), "[true, false]",
                                "[true, false, true, false]"},
        DeltaDictionaryTypeCase{arrow::int64(), "[1, -2]", "[1, -2, 3, 4]"},
        DeltaDictionaryTypeCase{arrow::int64(), "[1, null]", "[1, null, 3, 4]"},
        DeltaDictionaryTypeCase{arrow::uint64(), "[1, 2]", "[1, 2, 3, 4]"},
        DeltaDictionaryTypeCase{arrow::float64(), "[1.5, -2.25]",
                                "[1.5, -2.25, 3.5, 4.75]"},
        DeltaDictionaryTypeCase{arrow::utf8(), "[\"one\", \"two\"]",
                                "[\"one\", \"two\", \"three\", \"four\"]"},
        DeltaDictionaryTypeCase{arrow::binary(), "[\"one\", \"two\"]",
                                "[\"one\", \"two\", \"three\", \"four\"]"},
        DeltaDictionaryTypeCase{arrow::decimal128(10, 2), "[\"1.25\", \"-2.50\"]",
                                "[\"1.25\", \"-2.50\", \"3.75\", \"4.00\"]"},
        DeltaDictionaryTypeCase{arrow::list(arrow::int32()), "[[1, 2], []]",
                                "[[1, 2], [], [3, null], [4, 5]]"},
        DeltaDictionaryTypeCase{
            arrow::struct_({arrow::field("i", arrow::int32()),
                            arrow::field("s", arrow::utf8())}),
            "[{\"i\": 1, \"s\": \"one\"}, {\"i\": 2, \"s\": \"two\"}]",
            "[{\"i\": 1, \"s\": \"one\"}, {\"i\": 2, \"s\": \"two\"}, "
            "{\"i\": 3, \"s\": null}, {\"i\": 4, \"s\": \"four\"}]"},
        DeltaDictionaryTypeCase{arrow::fixed_size_list(arrow::int16(), 2),
                                "[[1, 2], [3, 4]]",
                                "[[1, 2], [3, 4], [5, 6], [7, 8]]"}));
#endif

// Build a struct array with a single dictionary-encoded (int32 -> utf8) child.
static void MakeDictionaryStructArray(struct ArrowArray* array,
                                      struct ArrowSchema* schema,
                                      const char* value0 = "foo",
                                      const char* value1 = "bar") {
  ASSERT_EQ(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(schema, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(schema->children[0], NANOARROW_TYPE_INT32),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(schema->children[0], "dict_col"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateDictionary(schema->children[0]), NANOARROW_OK);
  ASSERT_EQ(
      ArrowSchemaInitFromType(schema->children[0]->dictionary, NANOARROW_TYPE_STRING),
      NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(array, schema, nullptr), NANOARROW_OK);
  struct ArrowArray* indices = array->children[0];
  struct ArrowArray* values = indices->dictionary;

  ASSERT_EQ(ArrowArrayStartAppending(array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(values, ArrowCharView(value0)), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(values, ArrowCharView(value1)), NANOARROW_OK);

  ASSERT_EQ(ArrowArrayAppendInt(indices, 0), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(indices, 1), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(indices, 0), NANOARROW_OK);
  array->length = 3;

  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array, nullptr), NANOARROW_OK);
}

static std::vector<int32_t> DecodeMessageTypes(const struct ArrowBuffer* buffer) {
  std::vector<int32_t> message_types;
  struct ArrowBufferView remaining;
  remaining.data.as_uint8 = buffer->data;
  remaining.size_bytes = buffer->size_bytes;
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  ArrowIpcDecoderInit(&decoder);

  while (remaining.size_bytes > 0) {
    int result = ArrowIpcDecoderVerifyHeader(&decoder, remaining, &error);
    if (result == ENODATA) {
      break;
    }

    EXPECT_EQ(result, NANOARROW_OK) << error.message;
    if (result != NANOARROW_OK) {
      break;
    }

    message_types.push_back(decoder.message_type);
    int64_t message_size = ((decoder.header_size_bytes + 7) / 8) * 8 +
                           ((decoder.body_size_bytes + 7) / 8) * 8;
    EXPECT_LE(message_size, remaining.size_bytes);
    if (message_size > remaining.size_bytes) {
      break;
    }

    remaining.data.as_uint8 += message_size;
    remaining.size_bytes -= message_size;
  }

  ArrowIpcDecoderReset(&decoder);
  return message_types;
}

static std::vector<int64_t> DecodeDictionaryIds(const struct ArrowBuffer* buffer) {
  std::vector<int64_t> dictionary_ids;
  struct ArrowBufferView remaining;
  remaining.data.as_uint8 = buffer->data;
  remaining.size_bytes = buffer->size_bytes;
  struct ArrowIpcDecoder decoder;
  struct ArrowError error;
  ArrowIpcDecoderInit(&decoder);

  while (remaining.size_bytes > 0) {
    int result = ArrowIpcDecoderVerifyHeader(&decoder, remaining, &error);
    if (result == ENODATA) {
      break;
    }

    EXPECT_EQ(result, NANOARROW_OK) << error.message;
    if (result != NANOARROW_OK) {
      break;
    }

    if (decoder.message_type == NANOARROW_IPC_MESSAGE_TYPE_DICTIONARY_BATCH) {
      result = ArrowIpcDecoderDecodeHeader(&decoder, remaining, &error);
      EXPECT_EQ(result, NANOARROW_OK) << error.message;
      if (result != NANOARROW_OK) {
        break;
      }
      dictionary_ids.push_back(decoder.dictionary->id);
    }

    int64_t message_size = ((decoder.header_size_bytes + 7) / 8) * 8 +
                           ((decoder.body_size_bytes + 7) / 8) * 8;
    remaining.data.as_uint8 += message_size;
    remaining.size_bytes -= message_size;
  }

  ArrowIpcDecoderReset(&decoder);
  return dictionary_ids;
}

static void MakeNestedDictionaryStructArray(struct ArrowArray* array,
                                            struct ArrowSchema* schema,
                                            const char* inner_value1 = "bar") {
  ASSERT_EQ(ArrowSchemaInitFromType(schema, NANOARROW_TYPE_STRUCT), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(schema, 1), NANOARROW_OK);

  struct ArrowSchema* outer_field = schema->children[0];
  ASSERT_EQ(ArrowSchemaInitFromType(outer_field, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(outer_field, "outer"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateDictionary(outer_field), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(outer_field->dictionary, NANOARROW_TYPE_STRUCT),
            NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateChildren(outer_field->dictionary, 1), NANOARROW_OK);

  struct ArrowSchema* inner_field = outer_field->dictionary->children[0];
  ASSERT_EQ(ArrowSchemaInitFromType(inner_field, NANOARROW_TYPE_INT32), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaSetName(inner_field, "inner"), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaAllocateDictionary(inner_field), NANOARROW_OK);
  ASSERT_EQ(ArrowSchemaInitFromType(inner_field->dictionary, NANOARROW_TYPE_STRING),
            NANOARROW_OK);

  ASSERT_EQ(ArrowArrayInitFromSchema(array, schema, nullptr), NANOARROW_OK);
  struct ArrowArray* outer_indices = array->children[0];
  struct ArrowArray* outer_values = outer_indices->dictionary;
  struct ArrowArray* inner_indices = outer_values->children[0];
  struct ArrowArray* inner_values = inner_indices->dictionary;

  ASSERT_EQ(ArrowArrayStartAppending(array), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(inner_values, ArrowCharView("foo")), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(inner_values, ArrowCharView(inner_value1)),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(inner_indices, 0), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(inner_indices, 1), NANOARROW_OK);
  outer_values->length = 2;
  ASSERT_EQ(ArrowArrayAppendInt(outer_indices, 0), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendInt(outer_indices, 1), NANOARROW_OK);
  array->length = 2;
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(array, nullptr), NANOARROW_OK);
}

TEST(NanoarrowIpcWriter, WritesNestedDictionariesDependencyFirst) {
  struct ArrowError error;
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
  MakeNestedDictionaryStructArray(array.get(), schema.get());

  nanoarrow::UniqueArrayStream array_stream;
  ASSERT_EQ(ArrowBasicArrayStreamInit(array_stream.get(), schema.get(), 1), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(array_stream.get(), 0, array.get());

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayStream(writer.get(), array_stream.get(), &error),
            NANOARROW_OK)
      << error.message;

  EXPECT_EQ(DecodeDictionaryIds(output.get()), (std::vector<int64_t>{1, 0}));

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, output.get()), NANOARROW_OK);
  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(reader.get(), &input, nullptr), NANOARROW_OK);
  nanoarrow::UniqueSchema roundtrip_schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(reader.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  nanoarrow::UniqueArray roundtrip_array;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array.get(), &error),
            NANOARROW_OK)
      << error.message;

  struct ArrowArray* outer_dictionary = roundtrip_array->children[0]->dictionary;
  ASSERT_NE(outer_dictionary, nullptr);
  ASSERT_EQ(outer_dictionary->n_children, 1);
  struct ArrowArray* inner_dictionary = outer_dictionary->children[0]->dictionary;
  ASSERT_NE(inner_dictionary, nullptr);
  EXPECT_EQ(inner_dictionary->length, 2);
}

TEST(NanoarrowIpcWriter, ReemitsParentWhenNestedDictionaryChanges) {
  struct ArrowError error;
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array1;
  MakeNestedDictionaryStructArray(array1.get(), schema.get());

  nanoarrow::UniqueSchema unused_schema;
  nanoarrow::UniqueArray array2;
  MakeNestedDictionaryStructArray(array2.get(), unused_schema.get(), "baz");

  nanoarrow::UniqueArrayStream array_stream;
  ASSERT_EQ(ArrowBasicArrayStreamInit(array_stream.get(), schema.get(), 2), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(array_stream.get(), 0, array1.get());
  ArrowBasicArrayStreamSetArray(array_stream.get(), 1, array2.get());

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayStream(writer.get(), array_stream.get(), &error),
            NANOARROW_OK)
      << error.message;

  EXPECT_EQ(DecodeDictionaryIds(output.get()),
            (std::vector<int64_t>{1, 0, 1, 0}));

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, output.get()), NANOARROW_OK);
  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(reader.get(), &input, nullptr), NANOARROW_OK);
  nanoarrow::UniqueSchema roundtrip_schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(reader.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueArray roundtrip_array1;
  nanoarrow::UniqueArray roundtrip_array2;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array1.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array2.get(), &error),
            NANOARROW_OK)
      << error.message;

  const struct ArrowArray* inner_dictionary1 =
      roundtrip_array1->children[0]->dictionary->children[0]->dictionary;
  const struct ArrowArray* inner_dictionary2 =
      roundtrip_array2->children[0]->dictionary->children[0]->dictionary;
  nanoarrow::UniqueArrayView inner_view1;
  nanoarrow::UniqueArrayView inner_view2;
  ArrowArrayViewInitFromType(inner_view1.get(), NANOARROW_TYPE_STRING);
  ArrowArrayViewInitFromType(inner_view2.get(), NANOARROW_TYPE_STRING);
  ASSERT_EQ(ArrowArrayViewSetArray(inner_view1.get(), inner_dictionary1, &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(inner_view2.get(), inner_dictionary2, &error),
            NANOARROW_OK);
  EXPECT_EQ(ArrowArrayViewGetStringUnsafe(inner_view1.get(), 1), ArrowCharView("bar"));
  EXPECT_EQ(ArrowArrayViewGetStringUnsafe(inner_view2.get(), 1), ArrowCharView("baz"));
}

TEST(NanoarrowIpcWriter, DoesNotRepeatUnchangedDictionary) {
  struct ArrowError error;

  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array1;
  MakeDictionaryStructArray(array1.get(), schema.get());

  nanoarrow::UniqueSchema unused_schema;
  nanoarrow::UniqueArray array2;
  MakeDictionaryStructArray(array2.get(), unused_schema.get());

  nanoarrow::UniqueArrayStream array_stream;
  ASSERT_EQ(ArrowBasicArrayStreamInit(array_stream.get(), schema.get(), 2), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(array_stream.get(), 0, array1.get());
  ArrowBasicArrayStreamSetArray(array_stream.get(), 1, array2.get());

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayStream(writer.get(), array_stream.get(), &error),
            NANOARROW_OK)
      << error.message;

  std::vector<int32_t> message_types = DecodeMessageTypes(output.get());
  EXPECT_EQ(message_types,
            (std::vector<int32_t>{NANOARROW_IPC_MESSAGE_TYPE_SCHEMA,
                                  NANOARROW_IPC_MESSAGE_TYPE_DICTIONARY_BATCH,
                                  NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH,
                                  NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH}));
}

TEST(NanoarrowIpcWriter, EmitsChangedDictionary) {
  struct ArrowError error;

  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array1;
  MakeDictionaryStructArray(array1.get(), schema.get());

  nanoarrow::UniqueSchema unused_schema;
  nanoarrow::UniqueArray array2;
  MakeDictionaryStructArray(array2.get(), unused_schema.get(), "foo", "baz");

  nanoarrow::UniqueArrayStream array_stream;
  ASSERT_EQ(ArrowBasicArrayStreamInit(array_stream.get(), schema.get(), 2), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(array_stream.get(), 0, array1.get());
  ArrowBasicArrayStreamSetArray(array_stream.get(), 1, array2.get());

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayStream(writer.get(), array_stream.get(), &error),
            NANOARROW_OK)
      << error.message;

  std::vector<int32_t> message_types = DecodeMessageTypes(output.get());
  EXPECT_EQ(message_types,
            (std::vector<int32_t>{NANOARROW_IPC_MESSAGE_TYPE_SCHEMA,
                                  NANOARROW_IPC_MESSAGE_TYPE_DICTIONARY_BATCH,
                                  NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH,
                                  NANOARROW_IPC_MESSAGE_TYPE_DICTIONARY_BATCH,
                                  NANOARROW_IPC_MESSAGE_TYPE_RECORD_BATCH}));

#if defined(NANOARROW_BUILD_TESTS_WITH_ARROW)
  auto arrow_input = std::make_shared<arrow::io::BufferReader>(
      arrow::Buffer::Wrap(output->data, output->size_bytes));
  auto maybe_arrow_reader = arrow::ipc::RecordBatchStreamReader::Open(arrow_input);
  ASSERT_TRUE(maybe_arrow_reader.ok()) << maybe_arrow_reader.status();
  auto arrow_reader = maybe_arrow_reader.ValueUnsafe();

  std::shared_ptr<arrow::RecordBatch> arrow_batch1;
  std::shared_ptr<arrow::RecordBatch> arrow_batch2;
  ASSERT_TRUE(arrow_reader->ReadNext(&arrow_batch1).ok());
  ASSERT_TRUE(arrow_reader->ReadNext(&arrow_batch2).ok());
  auto arrow_dictionary1 =
      std::static_pointer_cast<arrow::DictionaryArray>(arrow_batch1->column(0));
  auto arrow_dictionary2 =
      std::static_pointer_cast<arrow::DictionaryArray>(arrow_batch2->column(0));
  auto arrow_values1 =
      std::static_pointer_cast<arrow::StringArray>(arrow_dictionary1->dictionary());
  auto arrow_values2 =
      std::static_pointer_cast<arrow::StringArray>(arrow_dictionary2->dictionary());
  EXPECT_EQ(arrow_values1->GetString(1), "bar");
  EXPECT_EQ(arrow_values2->GetString(1), "baz");
#endif

  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, output.get()), NANOARROW_OK);
  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(reader.get(), &input, nullptr), NANOARROW_OK);

  nanoarrow::UniqueSchema roundtrip_schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(reader.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueArray roundtrip_array1;
  nanoarrow::UniqueArray roundtrip_array2;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array1.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array2.get(), &error),
            NANOARROW_OK)
      << error.message;

  nanoarrow::UniqueArrayView roundtrip_view1;
  nanoarrow::UniqueArrayView roundtrip_view2;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(roundtrip_view1.get(), roundtrip_schema.get(),
                                        &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(roundtrip_view2.get(), roundtrip_schema.get(),
                                        &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(roundtrip_view1.get(), roundtrip_array1.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(roundtrip_view2.get(), roundtrip_array2.get(), &error),
            NANOARROW_OK)
      << error.message;

  struct ArrowStringView first_dictionary_value =
      ArrowArrayViewGetStringUnsafe(roundtrip_view1->children[0]->dictionary, 1);
  struct ArrowStringView replacement_dictionary_value =
      ArrowArrayViewGetStringUnsafe(roundtrip_view2->children[0]->dictionary, 1);
  EXPECT_EQ(std::string(first_dictionary_value.data,
                        first_dictionary_value.size_bytes),
            "bar");
  EXPECT_EQ(std::string(replacement_dictionary_value.data,
                        replacement_dictionary_value.size_bytes),
            "baz");

}

TEST(NanoarrowIpcWriter, RejectsChangedDictionaryInFile) {
  struct ArrowError error;
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array1;
  MakeDictionaryStructArray(array1.get(), schema.get());

  nanoarrow::UniqueSchema unused_schema;
  nanoarrow::UniqueArray array2;
  MakeDictionaryStructArray(array2.get(), unused_schema.get(), "foo", "baz");

  nanoarrow::UniqueArrayStream array_stream;
  ASSERT_EQ(ArrowBasicArrayStreamInit(array_stream.get(), schema.get(), 2), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(array_stream.get(), 0, array1.get());
  ArrowBasicArrayStreamSetArray(array_stream.get(), 1, array2.get());

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterStartFile(writer.get(), &error), NANOARROW_OK)
      << error.message;

  EXPECT_EQ(ArrowIpcWriterWriteArrayStream(writer.get(), array_stream.get(), &error),
            EINVAL);
  EXPECT_STREQ(error.message,
               "Arrow IPC files do not support replacement of dictionary ID 0");
}

TEST(NanoarrowIpcWriter, WritesDeltaDictionaryInFile) {
  struct ArrowError error;
  nanoarrow::UniqueSchema values_schema;
  ASSERT_EQ(ArrowSchemaInitFromType(values_schema.get(), NANOARROW_TYPE_STRING),
            NANOARROW_OK);
  nanoarrow::UniqueArray values;
  ASSERT_EQ(ArrowArrayInitFromSchema(values.get(), values_schema.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayStartAppending(values.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayAppendString(values.get(), ArrowCharView("delta")), NANOARROW_OK);
  ASSERT_EQ(ArrowArrayFinishBuildingDefault(values.get(), &error), NANOARROW_OK);
  nanoarrow::UniqueArrayView values_view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(values_view.get(), values_schema.get(), &error),
            NANOARROW_OK);
  ASSERT_EQ(ArrowArrayViewSetArray(values_view.get(), values.get(), &error),
            NANOARROW_OK);

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);
  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterStartFile(writer.get(), &error), NANOARROW_OK)
      << error.message;

  EXPECT_EQ(ArrowIpcWriterWriteDictionaryBatch(
                writer.get(), 0, /*is_delta=*/1, values_view.get(), &error),
            NANOARROW_OK)
      << error.message;
  auto* private_data =
      static_cast<struct ArrowIpcWriterPrivate*>(writer->private_data);
  EXPECT_EQ(private_data->footer.dictionary_blocks.size_bytes,
            sizeof(struct ArrowIpcFileBlock));
}

// Write a dictionary-encoded stream through the high-level WriteArrayStream path
// and read it back through the IPC reader, confirming the DictionaryBatch is
// emitted automatically and the decoded values match.
TEST(NanoarrowIpcWriter, RoundtripDictionaryStream) {
  struct ArrowError error;

  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
  MakeDictionaryStructArray(array.get(), schema.get());

  nanoarrow::UniqueArrayStream array_stream;
  ASSERT_EQ(ArrowBasicArrayStreamInit(array_stream.get(), schema.get(), 1), NANOARROW_OK);
  ArrowBasicArrayStreamSetArray(array_stream.get(), 0, array.get());

  nanoarrow::UniqueBuffer output;
  nanoarrow::ipc::UniqueOutputStream out_stream;
  ASSERT_EQ(ArrowIpcOutputStreamInitBuffer(out_stream.get(), output.get()), NANOARROW_OK);

  nanoarrow::ipc::UniqueWriter writer;
  ASSERT_EQ(ArrowIpcWriterInit(writer.get(), out_stream.get()), NANOARROW_OK);
  ASSERT_EQ(ArrowIpcWriterWriteArrayStream(writer.get(), array_stream.get(), &error),
            NANOARROW_OK)
      << error.message;

  // Read the encoded bytes back
  struct ArrowIpcInputStream input;
  ASSERT_EQ(ArrowIpcInputStreamInitBuffer(&input, output.get()), NANOARROW_OK);

  nanoarrow::UniqueArrayStream reader;
  ASSERT_EQ(ArrowIpcArrayStreamReaderInit(reader.get(), &input, nullptr), NANOARROW_OK);

  nanoarrow::UniqueSchema roundtrip_schema;
  ASSERT_EQ(ArrowArrayStreamGetSchema(reader.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(roundtrip_schema->n_children, 1);
  ASSERT_NE(roundtrip_schema->children[0]->dictionary, nullptr);
  EXPECT_STREQ(roundtrip_schema->children[0]->dictionary->format, "u");

  nanoarrow::UniqueArray roundtrip_array;
  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(roundtrip_array->length, 3);
  ASSERT_EQ(roundtrip_array->n_children, 1);
  ASSERT_NE(roundtrip_array->children[0]->dictionary, nullptr);
  EXPECT_EQ(roundtrip_array->children[0]->dictionary->length, 2);

  // Validate the decoded indices resolve to the original values
  nanoarrow::UniqueArrayView view;
  ASSERT_EQ(ArrowArrayViewInitFromSchema(view.get(), roundtrip_schema.get(), &error),
            NANOARROW_OK)
      << error.message;
  ASSERT_EQ(ArrowArrayViewSetArray(view.get(), roundtrip_array.get(), &error),
            NANOARROW_OK)
      << error.message;

  struct ArrowArrayView* indices_view = view->children[0];
  struct ArrowArrayView* values_view = indices_view->dictionary;
  ASSERT_NE(values_view, nullptr);
  EXPECT_EQ(ArrowArrayViewGetIntUnsafe(indices_view, 0), 0);
  EXPECT_EQ(ArrowArrayViewGetIntUnsafe(indices_view, 1), 1);
  EXPECT_EQ(ArrowArrayViewGetIntUnsafe(indices_view, 2), 0);

  struct ArrowStringView v0 = ArrowArrayViewGetStringUnsafe(values_view, 0);
  struct ArrowStringView v1 = ArrowArrayViewGetStringUnsafe(values_view, 1);
  EXPECT_EQ(std::string(v0.data, v0.size_bytes), "foo");
  EXPECT_EQ(std::string(v1.data, v1.size_bytes), "bar");

  ASSERT_EQ(ArrowArrayStreamGetNext(reader.get(), roundtrip_array.get(), &error),
            NANOARROW_OK)
      << error.message;
  EXPECT_EQ(roundtrip_array->release, nullptr);
}
