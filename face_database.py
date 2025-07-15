import face_recognition
import os
import numpy as np
from PIL import Image
import json
import base64
from io import BytesIO
import pickle
import requests
from urllib.parse import urlparse
import hashlib
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCloudFaceDatabase:
    def __init__(self, storage_type="github", fallback_storage=None, **kwargs):
        """
        Enhanced cloud face database with multi-platform support
        
        Args:
            storage_type: Primary storage ("github", "postgresql", "s3", "mongodb")
            fallback_storage: Fallback storage configuration
            **kwargs: Storage-specific configuration
        """
        self.known_faces = []
        self.known_labels = []
        self.face_metadata = []  # Store additional metadata
        self.storage_type = storage_type
        self.fallback_storage = fallback_storage
        self.config = kwargs
        self.last_sync = None
        self.auto_sync = kwargs.get('auto_sync', True)
        self.sync_interval = kwargs.get('sync_interval', 300)  # 5 minutes
        
        # Initialize primary storage
        self._init_storage(storage_type, kwargs)
        
        # Initialize fallback storage if provided
        if fallback_storage:
            self.fallback_config = fallback_storage
            self._init_fallback_storage()
        
        # Load existing data
        self.load_known_faces()
        
        # Set up auto-sync if enabled
        if self.auto_sync:
            self.schedule_sync()
    
    def _init_storage(self, storage_type, config):
        """Initialize storage based on type"""
        if storage_type == "github":
            self.init_github_storage(config)
        elif storage_type == "postgresql":
            self.init_postgresql(config)
        elif storage_type == "s3":
            self.init_s3_storage(config)
        elif storage_type == "mongodb":
            self.init_mongodb(config)
    
    def _init_fallback_storage(self):
        """Initialize fallback storage"""
        fallback_type = self.fallback_config.get('type')
        if fallback_type:
            self.fallback_storage_type = fallback_type
            self._init_storage(fallback_type, self.fallback_config)
    
    def init_github_storage(self, config):
        """Initialize GitHub Gist storage with enhanced features"""
        self.github_token = config.get('github_token') or os.environ.get('GITHUB_TOKEN')
        self.gist_id = config.get('gist_id') or os.environ.get('GIST_ID')
        self.gist_filename = config.get('gist_filename', 'face_database.json')
        self.backup_gist_filename = config.get('backup_gist_filename', 'face_database_backup.json')
        
        if not self.github_token:
            raise ValueError("GitHub token is required for GitHub storage")
        
        # Test connection
        self._test_github_connection()
    
    def _test_github_connection(self):
        """Test GitHub connection"""
        try:
            headers = {"Authorization": f"token {self.github_token}"}
            response = requests.get("https://api.github.com/user", headers=headers)
            if response.status_code == 200:
                logger.info("GitHub connection successful")
                return True
            else:
                logger.error(f"GitHub connection failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"GitHub connection error: {e}")
            return False
    
    def init_postgresql(self, config):
        """Initialize PostgreSQL with enhanced table structure"""
        try:
            import psycopg2
            from psycopg2.extras import Json
            
            self.db_url = config.get('database_url') or os.environ.get('DATABASE_URL')
            if not self.db_url:
                raise ValueError("Database URL is required for PostgreSQL storage")
            
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()
            
            # Enhanced table structure
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    id SERIAL PRIMARY KEY,
                    label VARCHAR(255) NOT NULL,
                    encoding BYTEA NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_faces_label ON faces(label)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_faces_created_at ON faces(created_at)
            ''')
            
            # Create sync log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sync_log (
                    id SERIAL PRIMARY KEY,
                    sync_type VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("PostgreSQL storage initialized successfully")
            
        except ImportError:
            raise ImportError("psycopg2 is required for PostgreSQL storage. Install with: pip install psycopg2-binary")
    
    def init_s3_storage(self, config):
        """Initialize Amazon S3 storage"""
        try:
            import boto3
            
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config.get('aws_access_key_id') or os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=config.get('aws_secret_access_key') or os.environ.get('AWS_SECRET_ACCESS_KEY'),
                region_name=config.get('aws_region', 'us-east-1')
            )
            self.bucket_name = config.get('bucket_name') or os.environ.get('S3_BUCKET_NAME')
            self.s3_key = config.get('s3_key', 'face_database.json')
            self.backup_s3_key = config.get('backup_s3_key', 'face_database_backup.json')
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info("S3 storage initialized successfully")
            
        except ImportError:
            raise ImportError("boto3 is required for S3 storage. Install with: pip install boto3")
    
    def init_mongodb(self, config):
        """Initialize MongoDB storage"""
        try:
            import pymongo
            
            self.mongo_url = config.get('mongo_url') or os.environ.get('MONGODB_URI')
            if not self.mongo_url:
                raise ValueError("MongoDB URL is required for MongoDB storage")
            
            self.client = pymongo.MongoClient(self.mongo_url)
            self.db = self.client.face_recognition
            self.collection = self.db.faces
            self.sync_collection = self.db.sync_log
            
            # Create indexes
            self.collection.create_index("label")
            self.collection.create_index("created_at")
            
            logger.info("MongoDB storage initialized successfully")
            
        except ImportError:
            raise ImportError("pymongo is required for MongoDB storage. Install with: pip install pymongo")
    
    def load_known_faces(self):
        """Load face data from primary storage with fallback"""
        try:
            success = self._load_from_primary()
            if not success and self.fallback_storage:
                logger.info("Primary storage failed, trying fallback storage")
                self._load_from_fallback()
            
            logger.info(f"Loaded {len(self.known_faces)} known faces")
            self.last_sync = datetime.now()
            
        except Exception as e:
            logger.error(f"Error loading known faces: {e}")
            if self.fallback_storage:
                logger.info("Attempting fallback storage")
                self._load_from_fallback()
    
    def _load_from_primary(self):
        """Load from primary storage"""
        try:
            if self.storage_type == "github":
                return self.load_from_github()
            elif self.storage_type == "postgresql":
                return self.load_from_postgresql()
            elif self.storage_type == "s3":
                return self.load_from_s3()
            elif self.storage_type == "mongodb":
                return self.load_from_mongodb()
            return False
        except Exception as e:
            logger.error(f"Primary storage load failed: {e}")
            return False
    
    def _load_from_fallback(self):
        """Load from fallback storage"""
        if not self.fallback_storage:
            return False
        
        try:
            fallback_type = self.fallback_storage.get('type')
            if fallback_type == "github":
                return self.load_from_github(use_fallback=True)
            elif fallback_type == "postgresql":
                return self.load_from_postgresql(use_fallback=True)
            elif fallback_type == "s3":
                return self.load_from_s3(use_fallback=True)
            elif fallback_type == "mongodb":
                return self.load_from_mongodb(use_fallback=True)
            return False
        except Exception as e:
            logger.error(f"Fallback storage load failed: {e}")
            return False
    
    def load_from_github(self, use_fallback=False):
        """Enhanced GitHub loading with backup support"""
        try:
            gist_id = self.gist_id
            token = self.github_token
            filename = self.gist_filename
            
            if use_fallback:
                gist_id = self.fallback_config.get('gist_id')
                token = self.fallback_config.get('github_token')
                filename = self.fallback_config.get('gist_filename', 'face_database.json')
            
            if not gist_id:
                logger.info("No gist_id provided, starting with empty database")
                return False
            
            url = f"https://api.github.com/gists/{gist_id}"
            headers = {"Authorization": f"token {token}"}
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                gist_data = response.json()
                
                # Try primary file first, then backup
                content = None
                if filename in gist_data['files']:
                    content = gist_data['files'][filename]['content']
                elif self.backup_gist_filename in gist_data['files']:
                    content = gist_data['files'][self.backup_gist_filename]['content']
                    logger.info("Using backup file from GitHub")
                
                if content:
                    data = json.loads(content)
                    self._load_data_from_json(data)
                    return True
            else:
                logger.error(f"Failed to load from GitHub: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading from GitHub: {e}")
            return False
    
    def load_from_postgresql(self, use_fallback=False):
        """Enhanced PostgreSQL loading"""
        try:
            import psycopg2
            
            db_url = self.db_url
            if use_fallback:
                db_url = self.fallback_config.get('database_url')
            
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            cursor.execute("SELECT label, encoding, metadata FROM faces ORDER BY created_at")
            rows = cursor.fetchall()
            
            for label, encoding_bytes, metadata in rows:
                encoding = pickle.loads(encoding_bytes)
                self.known_faces.append(encoding)
                self.known_labels.append(label)
                self.face_metadata.append(metadata or {})
            
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error loading from PostgreSQL: {e}")
            return False
    
    def load_from_s3(self, use_fallback=False):
        """Enhanced S3 loading with backup support"""
        try:
            s3_client = self.s3_client
            bucket_name = self.bucket_name
            s3_key = self.s3_key
            
            if use_fallback:
                # Configure fallback S3 client if needed
                pass
            
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
                content = response['Body'].read().decode('utf-8')
            except:
                # Try backup file
                response = s3_client.get_object(Bucket=bucket_name, Key=self.backup_s3_key)
                content = response['Body'].read().decode('utf-8')
                logger.info("Using backup file from S3")
            
            data = json.loads(content)
            self._load_data_from_json(data)
            return True
            
        except Exception as e:
            logger.error(f"Error loading from S3: {e}")
            return False
    
    def load_from_mongodb(self, use_fallback=False):
        """Enhanced MongoDB loading"""
        try:
            collection = self.collection
            if use_fallback:
                # Configure fallback MongoDB connection if needed
                pass
            
            for doc in collection.find().sort("created_at", 1):
                encoding = np.array(doc['encoding'])
                label = doc['label']
                metadata = doc.get('metadata', {})
                
                self.known_faces.append(encoding)
                self.known_labels.append(label)
                self.face_metadata.append(metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading from MongoDB: {e}")
            return False
    
    def _load_data_from_json(self, data):
        """Load data from JSON format"""
        for item in data:
            encoding = np.array(item['encoding'])
            label = item['label']
            metadata = item.get('metadata', {})
            
            self.known_faces.append(encoding)
            self.known_labels.append(label)
            self.face_metadata.append(metadata)
    
    def save_database(self, create_backup=True):
        """Save database to primary storage with backup"""
        try:
            # Save to primary storage
            primary_success = self._save_to_primary(create_backup)
            
            # Save to fallback storage if configured
            if self.fallback_storage:
                try:
                    self._save_to_fallback()
                except Exception as e:
                    logger.error(f"Fallback save failed: {e}")
            
            if primary_success:
                logger.info(f"Database saved to {self.storage_type} with {len(self.known_faces)} faces")
                self.last_sync = datetime.now()
                self._log_sync_event("save", "success")
            else:
                self._log_sync_event("save", "failed")
                
        except Exception as e:
            logger.error(f"Error saving database: {e}")
            self._log_sync_event("save", "error", str(e))
    
    def _save_to_primary(self, create_backup=True):
        """Save to primary storage"""
        try:
            if self.storage_type == "github":
                return self.save_to_github(create_backup)
            elif self.storage_type == "postgresql":
                return self.save_to_postgresql()
            elif self.storage_type == "s3":
                return self.save_to_s3(create_backup)
            elif self.storage_type == "mongodb":
                return self.save_to_mongodb()
            return False
        except Exception as e:
            logger.error(f"Primary storage save failed: {e}")
            return False
    
    def _save_to_fallback(self):
        """Save to fallback storage"""
        if not self.fallback_storage:
            return False
        
        fallback_type = self.fallback_storage.get('type')
        if fallback_type == "github":
            return self.save_to_github(use_fallback=True)
        elif fallback_type == "postgresql":
            return self.save_to_postgresql(use_fallback=True)
        elif fallback_type == "s3":
            return self.save_to_s3(use_fallback=True)
        elif fallback_type == "mongodb":
            return self.save_to_mongodb(use_fallback=True)
        return False
    
    def save_to_github(self, create_backup=True, use_fallback=False):
        """Enhanced GitHub saving with backup support"""
        try:
            data = []
            for i, encoding in enumerate(self.known_faces):
                data.append({
                    'label': self.known_labels[i],
                    'encoding': encoding.tolist(),
                    'metadata': self.face_metadata[i] if i < len(self.face_metadata) else {}
                })
            
            content = json.dumps(data, indent=2)
            
            gist_id = self.gist_id
            token = self.github_token
            filename = self.gist_filename
            
            if use_fallback:
                gist_id = self.fallback_config.get('gist_id')
                token = self.fallback_config.get('github_token')
                filename = self.fallback_config.get('gist_filename', 'face_database.json')
            
            files_payload = {
                filename: {"content": content}
            }
            
            # Add backup file if requested
            if create_backup:
                files_payload[self.backup_gist_filename] = {"content": content}
            
            if gist_id:
                # Update existing gist
                url = f"https://api.github.com/gists/{gist_id}"
                payload = {"files": files_payload}
                headers = {"Authorization": f"token {token}"}
                response = requests.patch(url, json=payload, headers=headers)
            else:
                # Create new gist
                url = "https://api.github.com/gists"
                payload = {
                    "description": "Face recognition database",
                    "public": False,
                    "files": files_payload
                }
                headers = {"Authorization": f"token {token}"}
                response = requests.post(url, json=payload, headers=headers)
                
                if response.status_code == 201:
                    if not use_fallback:
                        self.gist_id = response.json()['id']
                        logger.info(f"Created new gist: {self.gist_id}")
            
            return response.status_code in [200, 201]
            
        except Exception as e:
            logger.error(f"Error saving to GitHub: {e}")
            return False
    
    def save_to_postgresql(self, use_fallback=False):
        """Enhanced PostgreSQL saving"""
        try:
            import psycopg2
            from psycopg2.extras import Json
            
            db_url = self.db_url
            if use_fallback:
                db_url = self.fallback_config.get('database_url')
            
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("DELETE FROM faces")
            
            # Insert new data
            for i, encoding in enumerate(self.known_faces):
                encoding_bytes = pickle.dumps(encoding)
                metadata = self.face_metadata[i] if i < len(self.face_metadata) else {}
                
                cursor.execute(
                    "INSERT INTO faces (label, encoding, metadata) VALUES (%s, %s, %s)",
                    (self.known_labels[i], encoding_bytes, Json(metadata))
                )
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving to PostgreSQL: {e}")
            return False
    
    def save_to_s3(self, create_backup=True, use_fallback=False):
        """Enhanced S3 saving with backup support"""
        try:
            data = []
            for i, encoding in enumerate(self.known_faces):
                data.append({
                    'label': self.known_labels[i],
                    'encoding': encoding.tolist(),
                    'metadata': self.face_metadata[i] if i < len(self.face_metadata) else {}
                })
            
            content = json.dumps(data, indent=2)
            
            # Save main file
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.s3_key,
                Body=content,
                ContentType='application/json'
            )
            
            # Save backup file if requested
            if create_backup:
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=self.backup_s3_key,
                    Body=content,
                    ContentType='application/json'
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving to S3: {e}")
            return False
    
    def save_to_mongodb(self, use_fallback=False):
        """Enhanced MongoDB saving"""
        try:
            collection = self.collection
            if use_fallback:
                # Configure fallback MongoDB connection if needed
                pass
            
            # Clear existing data
            collection.delete_many({})
            
            # Insert new data
            documents = []
            for i, encoding in enumerate(self.known_faces):
                documents.append({
                    'label': self.known_labels[i],
                    'encoding': encoding.tolist(),
                    'metadata': self.face_metadata[i] if i < len(self.face_metadata) else {},
                    'created_at': datetime.now()
                })
            
            if documents:
                collection.insert_many(documents)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")
            return False
    
    def add_face(self, image_path, label, metadata=None):
        """Add new face with metadata"""
        try:
            # Load and process image
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                raise ValueError("No face detected in image")
            
            # Add to memory
            self.known_faces.append(encodings[0])
            self.known_labels.append(label)
            self.face_metadata.append(metadata or {
                'added_at': datetime.now().isoformat(),
                'source': 'web_upload'
            })
            
            # Save to cloud
            self.save_database()
            
            logger.info(f"Added face for label: {label}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add face: {e}")
            raise Exception(f"Failed to add face: {str(e)}")
    
    def predict(self, image_path, threshold=0.6):
        """Predict face with enhanced metadata"""
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                return None, 0, None, {}
            
            if not self.known_faces:
                return "unknown", 0, None, {}
            
            input_encoding = encodings[0]
            distances = face_recognition.face_distance(self.known_faces, input_encoding)
            
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]
            similarity = 1 - min_distance
            
            if similarity >= threshold:
                predicted_label = self.known_labels[min_distance_index]
                metadata = self.face_metadata[min_distance_index] if min_distance_index < len(self.face_metadata) else {}
                return predicted_label, similarity, min_distance, metadata
            else:
                return "unknown", similarity, min_distance, {}
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise Exception(f"Face prediction failed: {str(e)}")
    
    def sync_with_cloud(self, force=False):
        """Manually sync with cloud storage"""
        if not force and self.last_sync:
            time_diff = (datetime.now() - self.last_sync).seconds
            if time_diff < self.sync_interval:
                logger.info(f"Skipping sync, last sync was {time_diff} seconds ago")
                return
        
        logger.info("Syncing with cloud storage...")
        self.save_database()
    
    def schedule_sync(self):
        """Schedule automatic sync (implement with background task scheduler)"""
        # This would be implemented with celery, APScheduler, or similar
        pass
    
    def _log_sync_event(self, sync_type, status, details=None):
        """Log sync events"""
        log_entry = {
            'sync_type': sync_type,
            'status': status,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log to appropriate storage
        if self.storage_type == "postgresql":
            try:
                import psycopg2
                conn = psycopg2.connect(self.db_url)
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO sync_log (sync_type, status, details) VALUES (%s, %s, %s)",
                    (sync_type, status, details)
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"Failed to log sync event: {e}")
        elif self.storage_type == "mongodb":
            try:
                self.sync_collection.insert_one(log_entry)
            except Exception as e:
                logger.error(f"Failed to log sync event: {e}")
    
    def get_sync_status(self):
        """Get synchronization status"""
        return {
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'storage_type': self.storage_type,
            'fallback_configured': bool(self.fallback_storage),
            'auto_sync_enabled': self.auto_sync,
            'sync_interval': self.sync_interval
        }
    
    def get_stats(self):
        """Get enhanced database statistics"""
        stats = {
            'total_faces': len(self.known_faces),
            'unique_labels': len(set(self.known_labels)),
            'labels': dict([(label, self.known_labels.count(label)) for label in set(self.known_labels)]),
            'storage_type': self.storage_type,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'metadata_available': len(self.face_metadata) == len(self.known_faces)
        }
        return stats

# Configuration examples for different deployment scenarios
def get_github_railway_config():
    """Configuration for GitHub + Railway deployment"""
    return {
        'primary': {
            'storage_type': 'github',
            'github_token': os.environ.get('GITHUB_TOKEN'),
            'gist_id': os.environ.get('GIST_ID'),
            'auto_sync': True,
            'sync_interval': 300
        },
        'fallback': {
            'type': 'postgresql',
            'database_url': os.environ.get('DATABASE_URL')
        }
    }

def get_postgresql_s3_config():
    """Configuration for PostgreSQL + S3 backup"""
    return {
        'primary': {
            'storage_type': 'postgresql',
            'database_url': os.environ.get('DATABASE_URL'),
            'auto_sync': True,
            'sync_interval': 600
        },
        'fallback': {
            'type': 's3',
            'bucket_name': os.environ.get('S3_BUCKET_NAME'),
            'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY')
        }
    }

# Usage example
if __name__ == "__main__":
    # Initialize with GitHub as primary and PostgreSQL as fallback
    config = get_github_railway_config()
    
    db = EnhancedCloudFaceDatabase(
        storage_type=config['primary']['storage_type'],
        fallback_storage=config['fallback'],
        **config['primary']
    )
    
    print("Database initialized successfully!")
    print("Stats:", db.get_stats())
    print("Sync status:", db.get_sync_status())