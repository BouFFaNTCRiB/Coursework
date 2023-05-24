import os
import openai
import sys

if __name__ == "__main__":
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
    text = open(inp, 'r').read()
    openai.api_key = ''

    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=text,
      max_tokens=int(sys.argv[3]),
      temperature=0
    )
    output = open('out.xes', 'w')
    for line in response:
        output.write(f"{line}>\n")
