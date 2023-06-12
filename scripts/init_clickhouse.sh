#!/bin/bash
set -e

clickhouse client -n <<-EOSQL
	CREATE DATABASE datasets;
	CREATE TABLE datasets.openfood ENGINE=File(TSVRawWithNames, '/var/lib/clickhouse/user_files/en.openfoodfacts.org.products.csv');
	CREATE TABLE datasets.openfood_predictions (
	  prediction UInt32
	) ENGINE = MergeTree() ORDER BY tuple();
EOSQL