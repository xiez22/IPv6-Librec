# Generate the final output.
# Format: STU_ID;BOOK_ID1 BOOK_ID2 BOOK_ID3 ...

user_map_path = 'AFM/bslen15/user_map.txt'
book_map_path = 'new/sjtu_data/bind_book.txt'
data_path = 'new/sjtu_data/rank_output18.txt'
output_path = 'new/sjtu_data/final_output.txt'

print('Reading files...')
f_user_map = open(user_map_path, 'rt')
f_book_map = open(book_map_path, 'rt')
f_data = open(data_path, 'rt')
f_output = open(output_path, 'wt')

user_map = {}
book_map = {}

for line in f_user_map.readlines():
    user_id, code = line.strip().split(';')
    user_map[code] = user_id
f_user_map.close()

for line in f_book_map.readlines():
    book_id, _, _, code = line.strip().split(';')
    book_map[code] = book_id
f_book_map.close()

print('Writing...')
for line in f_data.readlines():
    user_code, _, _, rec_list = line.strip().split(';')
    user_id = user_map[user_code]
    rec_list = " ".join(map(book_map.get, rec_list.split()))
    
    f_output.write(f'{user_id};{rec_list}\n')
f_data.close()
f_output.close()

print('Finished!')
