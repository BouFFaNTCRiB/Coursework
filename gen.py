from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import gc

if __name__ == "__main__":
    # устанавливаем девайс обработки - если есть cuda и видеокарта, то gpu, иначе - cpu
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(sys.argv) < 4:
        print('wrong arg count')
        exit(1)
    inp = sys.argv[1]
    out = sys.argv[2]
    print(inp)
    if not inp.endswith('.xes'):
        print('wrong file type')
        exit(1)
    if not out.endswith('.xes'):
        print('wrong file type')
        exit(1)
    # обрабатываем входные данные
    text = open(inp, 'r').read()
    text = text.split('</log>', 1)[0]
    # инициализируем модель и токенизатор
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono", use_fast=False)
    # обрабатываем входные данные
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    # генерируем
    torch.cuda.empty_cache()
    gc.collect()
    generated_ids = model.generate(input_ids, max_new_tokens=int(sys.argv[3]))
    # выводим полученные данные в файл
    output = open('out.xes', 'w')
    i = 0
    line = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    line_e = line.replace("\n", "")
    lines = line_e.split('>')
    last = 0
    for lin in lines:
        if '</event' in lin:
            last = i
        i += 1
    for i in range(0, last):
        output.write(f"{lines[i]}>\n")
    output.write(f"\t</trace>\n")
    output.write(f"</log>\n")
