use ap;
SELECT invoice_number, invoice_date, invoice_total, (invoice_total - payment_total - credit_total) as 'balance_due'
FROM invoices
WHERE ((invoice_total - payment_total - credit_total) > 0 OR invoice_date > '2014-07-03')
ORDER BY invoice_date;

SELECT DATE_FORMAT(invoice_date, '%m/%d/%Y') as invoice_date, DATE_FORMAT(invoice_date, '%d-%m-%Y') as invoice_Date, round(invoice_total)
FROM invoices;


SELECT vendor_name, CONCAT(vendor_name, "'s Address") AS Vendor, CONCAT(vendor_city, CONCAT(', ', CONCAT(vendor_state,CONCAT(' ', vendor_zip_code)))) AS Address
FROM vendors;

SELECT count(*) as count_of_invoice, SUM(invoice_total - payment_total - credit_total) as Total_due
FROM invoices
WHERE vendor_id = 34;

SET SQL_SAFE_UPDATES = 0;

UPDATE invoices
SET credit_total = credit_total + 100
WHERE invoice_number = '97/522';

SELECT vendor_name, invoice_number, invoice_date, line_item_amount, account_description
FROM vendors v JOIN invoices i ON v.vendor_id = i.vendor_id JOIN invoice_line_items lt ON i.invoice_id = lt.invoice_id JOIN general_ledger_accounts g ON lt.account_number = g.account_number
WHERE (invoice_total - payment_total - credit_total) > 0
ORDER BY vendor_name, line_item_amount;
