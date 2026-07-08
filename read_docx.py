import zipfile
import xml.etree.ElementTree as ET

def read_docx(filename):
    with zipfile.ZipFile(filename) as docx:
        tree = ET.XML(docx.read('word/document.xml'))
        WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        PARA = WORD_NAMESPACE + 'p'
        TEXT = WORD_NAMESPACE + 't'
        
        text = ''
        for paragraph in tree.iter(PARA):
            for node in paragraph.iter(TEXT):
                if node.text:
                    text += node.text
            text += '\n'
        return text

if __name__ == '__main__':
    text = read_docx('GUARDIAN.docx')
    with open('guardian_text.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Success")
