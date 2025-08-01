import re
from num2hausa import number_to_hausa

weird_chars = ['‘', '।', '/', '\u200c', '\x94', '"', 
               '\u200d', '\x93' ,'…','“', '”', '_',  '\u200e',
               'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
               "«", "»", "(", ")", "+", "—", '\xad']

remove_weird_chars_pattern = re.compile('[' + ''.join(weird_chars) + ']', flags=re.UNICODE)

all_punc = ['!', '?', ',', ';', ':', '.']
all_punc_pattern = re.compile('[' + ''.join(all_punc) + ']', flags=re.UNICODE)
spaced_punc_pattern = re.compile('\w[' + ''.join(all_punc) + ']', flags=re.UNICODE)

newsent_punc = ['\?’', '\.’', '!’', '\?”', '\.”', '!”', '\?\"', '\."', '!"', '!', '\?', '\.']
newsent_punc_pattern = re.compile('(' + '|'.join(newsent_punc) + ')', flags=re.UNICODE)

spaced_punc = ['\?’', '\.’', '!’', '\?”', '\.”', '!”', '\?\"', '\."', '!"', '!', '\?', '\.', ';', ',']
spaced_punc_pattern = re.compile('(\s)(' + '|'.join(spaced_punc) + ')', flags=re.UNICODE)

def number_convert(text, language):
    text_norm = ""
    for l in text.split('\n'):
        line = ""
        le = re.sub(r'([0-9]+)(,| )([0-9]+)', r'\1\3', l) #join numericals with space inside
        for t in le.split():
            mlong = re.search(r'[0-9]+(,)[0-9]+', t) 
            mshort = re.search(r'[0-9]+', t) 
            m = mlong if mlong else mshort
            if m:
                num_term = t[m.start():m.end()].replace(',', '')
                if language == 'hausa':
                    num = number_to_hausa(int(num_term))
                elif language == 'luo':
                    num = ''
                elif language == 'chichewa':
                    num = None

                if num is not None:
                    textualized = t[0:m.start()] + num + t[m.end():]
                    line += textualized + " "
                else:
                    print("WARNING: can't convert %s (%s)"%(num_term, l))
                    if language == 'chichewa':
                        # keep number in quotes to prevent sentence splitting algorithm
                        # from treating it as a decimal at the end of a sentence
                        line += "'" + t + "' "
                    else:
                        line += t + " " # keep number
            else:
                line += t + " "
        line = line[:-1]
        text_norm += line + "\n"

    return text_norm


def normalize_text(text, language, newline_at_each_sent=False, remove_punc=False):
    clean_text = remove_weird_chars_pattern.sub(r'', text)
    if language == 'chichewa':
        # in chichewa, '’' is only used for quoting, so we can remove it
        clean_text = re.sub('’', '', clean_text)
    clean_text = re.sub('-', ' ', clean_text)
    clean_text = re.sub(r'\[|\]', '', clean_text)

    if newline_at_each_sent:
        clean_text = newsent_punc_pattern.sub(r'\1\n', clean_text)

    if remove_punc:
        clean_text = all_punc_pattern.sub(r'', clean_text)
        clean_text = clean_text.lower()

    else:
        clean_text = spaced_punc_pattern.sub(r'\2', clean_text)

    clean_text = number_convert(clean_text, language)

    clean_text = re.sub(' +', ' ', clean_text)
    clean_text = re.sub('\n +', '\n', clean_text)
    clean_text = re.sub('\n+', '\n', clean_text)
    clean_text = clean_text.strip()
    clean_text = re.sub('(’|"|”|\')+$', '', clean_text)

    return clean_text
