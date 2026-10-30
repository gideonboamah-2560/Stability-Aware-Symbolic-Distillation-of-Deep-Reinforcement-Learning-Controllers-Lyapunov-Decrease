"""Re-applies every Word fix: 12 pt Times New Roman styles, A4 page, full-width tables,
clean table alt text, Table 1 at 9 pt with sized columns and unsplittable rows."""
import re, sys, zipfile, shutil, os
src, dst = sys.argv[1], sys.argv[2]; d = "pp_tmp"; shutil.rmtree(d, ignore_errors=True); zipfile.ZipFile(src).extractall(d)
st = open(f"{d}/word/styles.xml", encoding="utf-8").read()
st = re.sub(r'<w:sz w:val="\d+"\s*/>', '<w:sz w:val="24" />', st); st = re.sub(r'<w:szCs w:val="\d+"\s*/>', '<w:szCs w:val="24" />', st)
for sid, v in [("Title", 32), ("Heading1", 28), ("Heading2", 26)]:
    m = re.search(r'(<w:style [^>]*w:styleId="%s".*?</w:style>)' % sid, st, re.S); blk = m.group(1)
    st = st.replace(blk, blk.replace('<w:sz w:val="24" />', '<w:sz w:val="%d" />' % v).replace('<w:szCs w:val="24" />', '<w:szCs w:val="%d" />' % v))
st = re.sub(r'<w:rFonts [^>]*/>', '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" w:cs="Times New Roman" />', st)
st = re.sub(r'<w:color [^>]*/>', '<w:color w:val="000000" />', st)
open(f"{d}/word/styles.xml", "w", encoding="utf-8").write(st)
x = open(f"{d}/word/document.xml", encoding="utf-8").read()
assert x.count("<w:sectPr />") == 1
x = x.replace("<w:sectPr />", '<w:sectPr><w:pgSz w:w="11906" w:h="16838" /><w:pgMar w:top="1418" w:right="1418" w:bottom="1418" w:left="1418" w:header="709" w:footer="709" w:gutter="0" /></w:sectPr>')
TW = 9070
WBY = {6: [0.8, 1.2, 1.3, 1.2, 0.8, 0.8], 8: [0.6, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0, 0.9], 9: [1.3, 1.0, 1.5, 1.3, 1.35, 1.25, 1.3, 1.25, 1.3], 5: [2.4, 1.5, 1.0, 1.0, 1.3]}
SZ = '<w:sz w:val="18" /><w:szCs w:val="18" />'
tables = list(re.finditer(r"<w:tbl>.*?</w:tbl>", x, re.S)); assert len(tables) == 4, len(tables); out, pos = [], 0
for ti, t in enumerate(tables):
    tb = t.group(0); nc = len(re.findall(r"<w:gridCol ", tb)); wts = WBY[nc]
    cols = [int(TW*w/sum(wts)) for w in wts]; cols[-1] = TW - sum(cols[:-1])
    tb = tb.replace('<w:tblW w:type="auto" w:w="0" />', '<w:tblW w:type="dxa" w:w="%d" />' % TW)
    it = iter(cols); tb = re.sub(r'<w:gridCol w:w="\d+"\s*/>', lambda m: '<w:gridCol w:w="%d" />' % next(it), tb)
    tb = re.sub(r"<w:tr>.*?</w:tr>|<w:tr .*?</w:tr>", lambda r: (lambda i2: re.sub(r'<w:tcW [^>]*/>', lambda m: '<w:tcW w:type="dxa" w:w="%d" />' % next(i2), r.group(0)))(iter(cols)), tb, flags=re.S)
    def clean(m):
        c = m.group(1)
        for a, b in [("\\pmstd", "±std"), ("\\pm", "±"), ("\\mathcal{D}", "D"), ("\\times", "×"), ("|\\Delta u|", "|Δu|"), ("\\Delta", "Δ")]: c = c.replace(a, b)
        return 'w:tblCaption w:val="%s"' % c
    tb = re.sub(r'w:tblCaption w:val="([^"]*)"', clean, tb)
    if nc >= 8:   # wide tables: 9 pt (8 pt for 9 columns), rows kept on one page
        SZ = '<w:sz w:val="16" /><w:szCs w:val="16" />' if nc == 9 else '<w:sz w:val="18" /><w:szCs w:val="18" />'
        tb = re.sub(r"<w:r>.*?</w:r>", lambda m: m.group(0).replace("<w:rPr>", "<w:rPr>" + SZ, 1) if "<w:rPr>" in m.group(0) else m.group(0).replace("<w:r>", "<w:r><w:rPr>" + SZ + "</w:rPr>", 1), tb, flags=re.S)
        tb = re.sub(r"<m:r>.*?</m:r>", lambda m: m.group(0) if "<w:rPr>" in m.group(0) else (m.group(0).replace("</m:rPr>", "</m:rPr><w:rPr>" + SZ + "</w:rPr>", 1) if "</m:rPr>" in m.group(0) else m.group(0).replace("<m:r>", "<m:r><w:rPr>" + SZ + "</w:rPr>", 1)), tb, flags=re.S)
        tb = re.sub(r"<w:trPr>", "<w:trPr><w:cantSplit />", tb); tb = re.sub(r"<w:tr>(?!<w:trPr>)", "<w:tr><w:trPr><w:cantSplit /></w:trPr>", tb)
    out.append(x[pos:t.start()] + tb); pos = t.end()
x = "".join(out) + x[pos:]
open(f"{d}/word/document.xml", "w", encoding="utf-8").write(x)
if os.path.exists(dst): os.remove(dst)
shutil.make_archive("pp_out", "zip", d); shutil.move("pp_out.zip", dst); print("postprocessed ->", dst)
