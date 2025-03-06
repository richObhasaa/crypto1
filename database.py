import sqlalchemy as sa
import pandas as pd
import logging
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Optional

from config import Config

# Create Base for declarative models
Base = declarative_base()

class CryptoDatabaseManager:
    """
    Comprehensive database management for cryptocurrency data
    """
    def __init__(self, db_type: str = None):
        """
        Initialize database connection
        
        Args:
            db_type: Type of database (postgresql, mongodb)
        """
        self.logger = logging.getLogger(__name__)
        
        # Database type selection
        self.db_type = db_type or Config.DATABASE_TYPE
        
        try:
            if self.db_type == 'postgresql':
                self._init_postgresql()
            elif self.db_type == 'mongodb':
                self._init_mongodb()
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
        except Exception as e:
            self.logger.error(f"Database initialization error: {e}")
            raise

    def _init_postgresql(self):
        """
        Initialize PostgreSQL connection
        """
        try:
            # Create SQLAlchemy engine
            self.engine = sa.create_engine(
                Config.DATABASE_URL, 
                pool_size=10, 
                max_overflow=20
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine
            )
            
            # Create tables
            Base.metadata.create_all(self.engine)
        except Exception as e:
            self.logger.error(f"PostgreSQL initialization error: {e}")
            raise

    def _init_mongodb(self):
        """
        Initialize MongoDB connection
        """
        try:
            import pymongo
            
            # Create MongoDB client
            self.client = pymongo.MongoClient(Config.DATABASE_URL)
            
            # Select database
            self.db = self.client.get_database('crypto_market')
        except ImportError:
            self.logger.error("PyMongo not installed")
            raise
        except Exception as e:
            self.logger.error(f"MongoDB initialization error: {e}")
            raise

    # PostgreSQL-specific methods
    def create_table(self, model_class):
        """
        Create a table for a given SQLAlchemy model
        
        Args:
            model_class: SQLAlchemy model class
        """
        if self.db_type != 'postgresql':
            raise NotImplementedError("Method only supported for PostgreSQL")
        
        try:
            model_class.__table__.create(self.engine)
        except Exception as e:
            self.logger.error(f"Table creation error: {e}")

    def insert_data_postgresql(
        self, 
        table_name: str, 
        data: List[Dict]
    ) -> int:
        """
        Insert data into PostgreSQL table
        
        Args:
            table_name: Name of the table
            data: List of dictionaries to insert
        
        Returns:
            Number of rows inserted
        """
        if self.db_type != 'postgresql':
            raise NotImplementedError("Method only supported for PostgreSQL")
        
        session = self.SessionLocal()
        try:
            # Dynamic table lookup
            table = Base.metadata.tables[table_name]
            
            # Bulk insert
            session.execute(table.insert(), data)
            session.commit()
            
            return len(data)
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"PostgreSQL insert error: {e}")
            return 0
        finally:
            session.close()

    # MongoDB-specific methods
    def insert_data_mongodb(
        self, 
        collection_name: str, 
        data: List[Dict]
    ) -> int:
        """
        Insert data into MongoDB collection
        
        Args:
            collection_name: Name of the collection
            data: List of documents to insert
        
        Returns:
            Number of documents inserted
        """
        if self.db_type != 'mongodb':
            raise NotImplementedError("Method only supported for MongoDB")
        
        try:
            collection = self.db[collection_name]
            result = collection.insert_many(data)
            return len(result.inserted_ids)
        except Exception as e:
            self.logger.error(f"MongoDB insert error: {e}")
            return 0

    def query_data(
        self, 
        table_or_collection: str, 
        filters: Dict = None, 
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Query data from database
        
        Args:
            table_or_collection: Name of table/collection
            filters: Query filters
            limit: Maximum number of records
        
        Returns:
            DataFrame with query results
        """
        filters = filters or {}
        
        try:
            if self.db_type == 'postgresql':
                # SQLAlchemy query
                session = self.SessionLocal()
                query = session.query(table_or_collection).filter_by(**filters).limit(limit)
                result = pd.read_sql(query.statement, session.bind)
                session.close()
                return result
            
            elif self.db_type == 'mongodb':
                # MongoDB query
                collection = self.db[table_or_collection]
                cursor = collection.find(filters).limit(limit)
                result = pd.DataFrame(list(cursor))
                return result
        
        except Exception as e:
            self.logger.error(f"Data query error: {e}")
            return pd.DataFrame()

    def upsert_data(
        self, 
        table_or_collection: str, 
        data: List[Dict], 
        unique_key: str = 'id'
    ) -> int:
        """
        Upsert (insert or update) data
        
        Args:
            table_or_collection: Name of table/collection
            data: List of documents/records
            unique_key: Field to use for identifying unique records
        
        Returns:
            Number of records upserted
        """
        try:
            if self.db_type == 'postgresql':
                # PostgreSQL upsert (using ON CONFLICT)
                session = self.SessionLocal()
                upserted_count = 0
                
                for record in data:
                    stmt = sa.dialects.postgresql.insert(
                        Base.metadata.tables[table_or_collection]
                    ).values(**record)
                    
                    upsert_stmt = stmt.on_conflict_do_update(
                        index_elements=[unique_key],
                        set_=record
                    )
                    
                    session.execute(upsert_stmt)
                    upserted_count += 1
                
                session.commit()
                session.close()
                return upserted_count
        
        except Exception as e:
            self.logger.error(f"Upsert operation error: {e}")
            return 0

    def delete_data(
        self, 
        table_or_collection: str, 
        filters: Dict
    ) -> int:
        """
        Delete data from database
        
        Args:
            table_or_collection: Name of table/collection
            filters: Deletion filters
        
        Returns:
            Number of records deleted
        """
        try:
            if self.db_type == 'postgresql':
                session = self.SessionLocal()
                deleted_count = session.query(
                    Base.metadata.tables[table_or_collection]
                ).filter_by(**filters).delete()
                session.commit()
                session.close()
                return deleted_count
            
            elif self.db_type == 'mongodb':
                collection = self.db[table_or_collection]
                result = collection.delete_many(filters)
                return result.deleted_count
        
        except Exception as e:
            self.logger.error(f"Data deletion error: {e}")
            return 0

    def backup_data(
        self, 
        table_or_collection: str, 
        backup_file: str
    ) -> bool:
        """
        Backup data to a file
        
        Args:
            table_or_collection: Name of table/collection
            backup_file: Path to backup file
        
        Returns:
            Backup success status
        """
        try:
            # Query data
            data = self.query_data(table_or_collection)
            
            # Determine backup format based on file extension
            if backup_file.endswith('.csv'):
                data.to_csv(backup_file, index=False)
            elif backup_file.endswith('.json'):
                data.to_json(backup_file, orient='records')
            elif backup_file.endswith('.parquet'):
                data.to_parquet(backup_file)
            else:
                raise ValueError("Unsupported backup format")
            
            self.logger.info(f"Backup created: {backup_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Backup creation error: {e}")
            return False

    def restore_data(
        self, 
        table_or_collection: str, 
        backup_file: str
    ) -> int:
        """
        Restore data from a backup file
        
        Args:
            table_or_collection: Name of table/collection
            backup_file: Path to backup file
        
        Returns:
            Number of records restored
        """
        try:
            # Read backup file
            if backup_file.endswith('.csv'):
                data = pd.read_csv(backup_file)
            elif backup_file.endswith('.json'):
                data = pd.read_json(backup_file, orient='records')
            elif backup_file.endswith('.parquet'):
                data = pd.read_parquet(backup_file)
            else:
                raise ValueError("Unsupported backup format")
            
            # Convert to list of dictionaries
            data_records = data.to_dict('records')
            
            # Upsert data
            restored_count = self.upsert_data(table_or_collection, data_records)
            
            self.logger.info(f"Restored {restored_count} records from {backup_file}")
            return restored_count
        
        except Exception as e:
            self.logger.error(f"Data restoration error: {e}")
            return 0

    # Database models for cryptocurrency data
    class CryptoPrice(Base):
        """
        SQLAlchemy model for storing cryptocurrency prices
        """
        __tablename__ = 'crypto_prices'
        
        id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
        token = sa.Column(sa.String, nullable=False)
        price = sa.Column(sa.Float, nullable=False)
        timestamp = sa.Column(sa.DateTime, nullable=False)
        market_cap = sa.Column(sa.Float)
        volume = sa.Column(sa.Float)

    class CryptoTrend(Base):
        """
        SQLAlchemy model for storing cryptocurrency trends
        """
        __tablename__ = 'crypto_trends'
        
        id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
        token = sa.Column(sa.String, nullable=False)
        source = sa.Column(sa.String)  # Twitter, Reddit, News
        sentiment = sa.Column(sa.Float)
        engagement = sa.Column(sa.Integer)
        timestamp = sa.Column(sa.DateTime, nullable=False)

def main():
    """
    Test the CryptoDatabaseManager
    """
    logging.basicConfig(level=logging.INFO)
    
    # Test PostgreSQL
    print("Testing PostgreSQL Database Manager")
    pg_db = CryptoDatabaseManager(db_type='postgresql')
    
    # Sample cryptocurrency price data
    sample_prices = [
        {
            'token': 'bitcoin',
            'price': 50000,
            'timestamp': pd.Timestamp.now(),
            'market_cap': 1000000000,
            'volume': 5000000
        },
        {
            'token': 'ethereum',
            'price': 3500,
            'timestamp': pd.Timestamp.now(),
            'market_cap': 400000000,
            'volume': 2000000
        }
    ]
    
    # Insert data
    insert_count = pg_db.insert_data_postgresql('crypto_prices', sample_prices)
    print(f"Inserted {insert_count} records")
    
    # Query data
    query_result = pg_db.query_data('crypto_prices')
    print("\nQuery Result:")
    print(query_result)
    
    # Backup data
    backup_success = pg_db.backup_data('crypto_prices', 'crypto_prices_backup.csv')
    print(f"\nBackup Success: {backup_success}")
    
    # MongoDB testing would be similar
    try:
        print("\nTesting MongoDB Database Manager")
        mongo_db = CryptoDatabaseManager(db_type='mongodb')
        # Add MongoDB-specific tests here
    except Exception as e:
        print(f"MongoDB testing skipped: {e}")

if __name__ == "__main__":
    main()count
            
            elif self.db_type == 'mongodb':
                # MongoDB upsert
                collection = self.db[table_or_collection]
                upserted_count = 0
                
                for record in data:
                    collection.update_one(
                        {unique_key: record[unique_key]},
                        {'$set': record},
                        upsert=True
                    )
                    upserted_count += 1
                
                return upserted_