#m.py
import ir_datasets
from tqdm import tqdm
from database import setup_documents_table, upload_docs_with_cleaning, get_existing_doc_ids

def process_dataset(dataset_path, table_name):
    print(f"⬆️ تحميل مجموعة {dataset_path}...")
    ds = ir_datasets.load(dataset_path)

    count = 0
    for doc in tqdm(ds.docs_iter(), desc=f"🧾 قراءة مستندات {dataset_path}"):
        try:
            _ = doc.text
            count += 1
        except UnicodeDecodeError:
            print(f"⚠️ تخطي مستند لوجود خطأ ترميز: {doc.doc_id}")

    print(f"📄 عدد المستندات المقروءة بنجاح في {dataset_path}: {count}")
    #لرفع المستندات المنظفة لمجموعة البيانات على الجدول الخاص فيها
    upload_docs_with_cleaning(table_name, ds)

def main():
    print("🚀 <<< Start >>>")

    datasets = [
        ("beir/quora/dev", "documents_beir_quora_dev"),
        ("antique/test", "documents_antique_test")
    ]
    
    for dataset_path, table_name in datasets:
        print(f"🔧 إعداد جدول قاعدة البيانات: {table_name}")

        #انشاء جدول في قاعدة البيانات لهذه المجموعة 
        setup_documents_table(table_name)

        #الحصول على معرفات المستندات الموجودة مسبقا في الجدول 
        existing_ids = get_existing_doc_ids(table_name)
        if existing_ids:
            #هنا نتخطى عملية التحميل في حال كان يوجد تكرار 
            print(f"⏭️ وجدنا {len(existing_ids)} مستند في {table_name}، لذلك لن نقوم بإعادة التحميل.")
        else:
            #نبدأ بتحميل البيانات 
            process_dataset(dataset_path, table_name)

if __name__ == "__main__":
    main()
