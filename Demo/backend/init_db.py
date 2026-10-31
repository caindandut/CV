"""Create the configured MySQL database and application tables."""

import os
import sys
import re
import pymysql
from app import config  # noqa: F401 -- load Demo/.env before constructing the DB URL
from app.database import Base, engine


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    name = os.getenv("DB_NAME", "helmet_demo")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", name):
        raise SystemExit("DB_NAME chỉ được chứa chữ, số và dấu gạch dưới")
    try:
        connection = pymysql.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "3306")),
            user=os.getenv("DB_USER", "helmet_demo"),
            password=os.getenv("DB_PASSWORD", ""),
            charset="utf8mb4",
            connect_timeout=3,
        )
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE DATABASE IF NOT EXISTS `{name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
        Base.metadata.create_all(engine)
        print("Đã tạo database và bảng jobs")
    except Exception:
        raise SystemExit(
            "Không tạo được database/bảng. Kiểm tra DB_* trong .env và quyền CREATE của tài khoản MySQL."
        )


if __name__ == "__main__":
    main()
