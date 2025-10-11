import pytest

from promise_aware_eta.data_ingestion import TABLE_FILENAMES


@pytest.fixture()
def sample_raw_dir(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / TABLE_FILENAMES["orders"]).write_text(
        "order_id,customer_id,order_purchase_timestamp,order_approved_at,order_delivered_carrier_date,order_delivered_customer_date,order_estimated_delivery_date\n"
        "order_1,cust_1,2017-01-01 10:00:00,2017-01-01 10:15:00,2017-01-02 04:00:00,2017-01-05 12:00:00,2017-01-06 00:00:00\n"
        "order_2,cust_2,2017-01-02 09:30:00,2017-01-02 10:00:00,2017-01-03 06:00:00,2017-01-08 15:00:00,2017-01-07 00:00:00\n"
    )

    (raw_dir / TABLE_FILENAMES["order_items"]).write_text(
        "order_id,order_item_id,product_id,seller_id,shipping_limit_date,price,freight_value\n"
        "order_1,1,prod_1,seller_1,2017-01-02 00:00:00,100,10\n"
        "order_2,1,prod_2,seller_2,2017-01-03 00:00:00,150,12\n"
    )

    (raw_dir / TABLE_FILENAMES["order_payments"]).write_text(
        "order_id,payment_sequential,payment_type,payment_installments,payment_value\n"
        "order_1,1,credit_card,2,110\n"
        "order_2,1,boleto,1,162\n"
    )

    (raw_dir / TABLE_FILENAMES["order_reviews"]).write_text(
        "review_id,order_id,review_score,review_creation_date,review_answer_timestamp\n"
        "rev_1,order_1,5,2017-01-10 00:00:00,2017-01-10 05:00:00\n"
        "rev_2,order_2,2,2017-01-12 00:00:00,2017-01-12 07:00:00\n"
    )

    (raw_dir / TABLE_FILENAMES["products"]).write_text(
        "product_id,product_category_name,product_weight_g,product_length_cm\n"
        "prod_1,category_a,1000,20\n"
        "prod_2,category_b,800,25\n"
    )

    (raw_dir / TABLE_FILENAMES["sellers"]).write_text(
        "seller_id,seller_zip_code_prefix,seller_city,seller_state\n"
        "seller_1,12345,CityA,SP\n"
        "seller_2,54321,CityB,RJ\n"
    )

    (raw_dir / TABLE_FILENAMES["customers"]).write_text(
        "customer_id,customer_unique_id,customer_zip_code_prefix,customer_city,customer_state\n"
        "cust_1,u1,11111,CityA,SP\n"
        "cust_2,u2,22222,CityC,MG\n"
    )

    (raw_dir / TABLE_FILENAMES["geolocation"]).write_text(
        "geolocation_zip_code_prefix,geolocation_lat,geolocation_lng\n"
        "11111,-23.5,-46.6\n"
        "22222,-22.9,-43.2\n"
    )

    (raw_dir / TABLE_FILENAMES["product_category_translation"]).write_text(
        "product_category_name,product_category_name_english\n"
        "category_a,Category A\n"
        "category_b,Category B\n"
    )

    return raw_dir
