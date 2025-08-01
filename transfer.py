import csv
import json

def convert_csv_to_json(csv_filepath, output_json_filepath):
    """
    Reads a CSV file with DNA sequences and labels, converts each row
    to a specified JSON format, and writes each JSON object to a new line
    in the output JSON file.

    Args:
        csv_filepath (str): The path to the input CSV file.
        output_json_filepath (str): The path to the output JSON file.
    """
    instruction_text = "Determine if the following DNA sequence is a promoter or a non-promoter ."
    converted_count = 0
    skipped_count = 0

    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as csvfile, \
             open(output_json_filepath, mode='w', encoding='utf-8') as outfile:

            reader = csv.reader(csvfile)
            try:
                header = next(reader) # 跳過標頭行
                if header != ['sequence', 'label']:
                    print(f"警告：CSV 檔案標頭不是預期的 ['sequence', 'label']，而是 {header}。仍會嘗試處理。")
            except StopIteration:
                print(f"錯誤：CSV 檔案 '{csv_filepath}' 為空或只有標頭。")
                return

            for row_number, row in enumerate(reader, 1):
                if len(row) == 2:
                    sequence, label_str = row

                    if label_str == '1':
                        output_label = "promoter"
                    elif label_str == '0':
                        output_label = "Non-promoter"
                    else:
                        print(f"警告：CSV檔案第 {row_number+1} 行（資料行 {row_number}）的標籤 '{label_str}' 不是 '0' 或 '1'。此行將被跳過。")
                        skipped_count += 1
                        continue

                    json_object = {
                        "instruction": instruction_text,
                        "input": sequence,
                        "output": output_label
                    }
                    # 將 JSON 物件轉換為字串並寫入檔案，每個 JSON 物件佔一行
                    outfile.write(json.dumps(json_object, ensure_ascii=False) + '\n')
                    converted_count += 1
                else:
                    print(f"警告：CSV檔案第 {row_number+1} 行（資料行 {row_number}）的格式不正確（應有2個欄位，實際為 {len(row)} 個）。此行將被跳過：{row}")
                    skipped_count += 1

            print(f"轉換完成。成功轉換 {converted_count} 行，跳過 {skipped_count} 行。")
            print(f"結果已儲存到 '{output_json_filepath}'。")

    except FileNotFoundError:
        print(f"錯誤：找不到輸入檔案 '{csv_filepath}'。請確認檔案名稱和路徑是否正確。")
    except Exception as e:
        print(f"處理過程中發生錯誤：{e}")

if __name__ == "__main__":
    csv_input_file = "train.csv"      # 輸入的CSV檔案名稱
    json_output_file = "train.json"   # 輸出的JSON檔案名稱

    convert_csv_to_json(csv_input_file, json_output_file)
