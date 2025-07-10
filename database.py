#database.py
import mysql.connector
from text_cleaner import clean_text  # تأكد من وجود دالة تنظيف النص في هذا الملف

# 1. فتح اتصال بقاعدة البيانات MySQL

def open_connection():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='Search_enginee'
    )
    return conn, conn.cursor()

# 2. إنشاء جدول خاص بكل مجموعة بيانات (كل جدول باسم documents_<اسم_المجموعة>)
def setup_documents_table(table_name):
    conn, cursor = open_connection()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            doc_id VARCHAR(255) PRIMARY KEY,
            title TEXT,
            content LONGTEXT,
            cleaned_content LONGTEXT
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

# 3. إدخال دفعة من المستندات (batch insert) مع تحديث السجلات إذا تكررت
def insert_batch(buffer, table_name):
    conn, cursor = open_connection()
    sql = f"""
        INSERT INTO {table_name} (doc_id, title, content, cleaned_content)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            title=VALUES(title),
            content=VALUES(content),
            cleaned_content=VALUES(cleaned_content)
    """
    cursor.executemany(sql, buffer)  # استخدم executemany لتحسين الأداء مع دفعات كبيرة
    conn.commit()
    cursor.close()
    conn.close()

# 4. جلب جميع معرفات المستندات الموجودة مسبقاً في الجدول لتجنب التكرار
def get_existing_doc_ids(table_name):
    conn, cursor = open_connection()
    cursor.execute(f"SELECT doc_id FROM {table_name}")
    ids = {row[0] for row in cursor.fetchall()}
    cursor.close()
    conn.close()
    return ids

BATCH_SIZE = 200  # أو 100 حسب الحاجة

def upload_docs_with_cleaning(table_name, dataset_obj):
    existing_ids = get_existing_doc_ids(table_name)
    count = 0
    skipped = 0
    buffer = []

    for doc in dataset_obj.docs_iter():
        if doc.doc_id in existing_ids:
            skipped += 1
            continue

        try:
            content = getattr(doc, 'text', '') or getattr(doc, 'content', '') or ''
            _ = content.encode('utf-8', errors='strict')
            doc_id = doc.doc_id
            title = getattr(doc, 'title', '') or ''
            cleaned = clean_text(content)

            buffer.append((doc_id, title, content, cleaned))
            count += 1

            if count % BATCH_SIZE == 0:
                insert_batch(buffer, table_name)
                print(f"[{table_name}] ✅ تم إدخال {count} مستندات...")
                buffer = []

        except UnicodeDecodeError:
            print(f"⚠️ تخطي مستند به مشكلة ترميز: {doc.doc_id}")

    if buffer:
        insert_batch(buffer, table_name)

    print(f"✅ [{table_name}] تم إدخال {count} مستند جديد.")
    print(f"⏭️ [{table_name}] تم تجاوز {skipped} مستند مكرر.")


# 6. جلب جميع المستندات المنظفة من جدول معين (doc_id والنص المنظف)
def fetch_cleaned_documents(table_name):
    conn, cursor = open_connection()
    cursor.execute(f"SELECT doc_id, cleaned_content FROM {table_name}")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    doc_ids = [row[0] for row in rows]
    texts = [row[1] for row in rows]
    return doc_ids, texts

#database.py
import mysql.connector
from text_cleaner import clean_text  # تأكد من وجود دالة تنظيف النص في هذا الملف

# 1. فتح اتصال بقاعدة البيانات MySQL

def open_connection():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='Search_enginee'
    )
    return conn, conn.cursor()

# 2. إنشاء جدول خاص بكل مجموعة بيانات (كل جدول باسم documents_<اسم_المجموعة>)
def setup_documents_table(table_name):
    conn, cursor = open_connection()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            doc_id VARCHAR(255) PRIMARY KEY,
            title TEXT,
            content LONGTEXT,
            cleaned_content LONGTEXT
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

# 3. إدخال دفعة من المستندات (batch insert) مع تحديث السجلات إذا تكررت
def insert_batch(buffer, table_name):
    conn, cursor = open_connection()
    sql = f"""
        INSERT INTO {table_name} (doc_id, title, content, cleaned_content)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            title=VALUES(title),
            content=VALUES(content),
            cleaned_content=VALUES(cleaned_content)
    """
    cursor.executemany(sql, buffer)  # استخدم executemany لتحسين الأداء مع دفعات كبيرة
    conn.commit()
    cursor.close()
    conn.close()

# 4. جلب جميع معرفات المستندات الموجودة مسبقاً في الجدول لتجنب التكرار
def get_existing_doc_ids(table_name):
    conn, cursor = open_connection()
    cursor.execute(f"SELECT doc_id FROM {table_name}")
    ids = {row[0] for row in cursor.fetchall()}
    cursor.close()
    conn.close()
    return ids

BATCH_SIZE = 200  # أو 100 حسب الحاجة

def upload_docs_with_cleaning(table_name, dataset_obj):
    existing_ids = get_existing_doc_ids(table_name)
    count = 0
    skipped = 0
    buffer = []

    for doc in dataset_obj.docs_iter():
        if doc.doc_id in existing_ids:
            skipped += 1
            continue

        try:
            content = getattr(doc, 'text', '') or getattr(doc, 'content', '') or ''
            _ = content.encode('utf-8', errors='strict')
            doc_id = doc.doc_id
            title = getattr(doc, 'title', '') or ''
            cleaned = clean_text(content)

            buffer.append((doc_id, title, content, cleaned))
            count += 1

            if count % BATCH_SIZE == 0:
                insert_batch(buffer, table_name)
                print(f"[{table_name}] ✅ تم إدخال {count} مستندات...")
                buffer = []

        except UnicodeDecodeError:
            print(f"⚠️ تخطي مستند به مشكلة ترميز: {doc.doc_id}")

    if buffer:
        insert_batch(buffer, table_name)

    print(f"✅ [{table_name}] تم إدخال {count} مستند جديد.")
    print(f"⏭️ [{table_name}] تم تجاوز {skipped} مستند مكرر.")


# 6. جلب جميع المستندات المنظفة من جدول معين (doc_id والنص المنظف)
def fetch_cleaned_documents(table_name):
    conn, cursor = open_connection()
    cursor.execute(f"SELECT doc_id, cleaned_content FROM {table_name}")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    doc_ids = [row[0] for row in rows]
    texts = [row[1] for row in rows]
    return doc_ids, texts

# def fetch_cleaned_documents_by_table(table_name):
#     conn, cursor = open_connection()
#     cursor.execute(f"SELECT doc_id, cleaned_content FROM {table_name}")
#     rows = cursor.fetchall()
#     cursor.close()
#     conn.close()

#     doc_ids = [row[0] for row in rows]
#     texts = [row[1] for row in rows]
#     return doc_ids, texts
