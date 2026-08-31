from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Internship\CCNA-Exam-Prep-Guide.pdf")
data = loader.load()

full_text = "\n\n".join(page.page_content for page in data)
print(full_text[:])