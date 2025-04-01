from pypdf import PdfWriter
import os

in_dir = "mdsap"
out_dir = "mdsap/merged"
out_filename = "merged2.pdf"

if not(os.path.exists(out_dir)):
  os.makedirs(out_dir)

files = os.listdir(in_dir)
pdf_files = [(in_dir + "/" + f) for f in files if f.endswith(".pdf")]
pdf_files

merger = PdfWriter()

for pdf in pdf_files:
    merger.append(pdf)

merger.write(out_dir + "/" + out_filename)
merger.close()

