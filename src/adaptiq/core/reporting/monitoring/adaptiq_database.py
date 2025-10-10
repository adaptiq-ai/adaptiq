import logging
import redis
from redis import Redis
from typing import Optional

from adaptiq.core.entities import QTablePayload, DatabaseConfig


class DatabaseManager:
    """
    Manager for Redis database operations with logging.

    This class provides a unified interface for storing and retrieving Q-Table data
    from Redis Cloud. It handles connection management, serialization/deserialization
    of QTablePayload objects, and provides comprehensive logging for all operations.

    Attributes:
        database_host (str): Redis server hostname
        database_port (int): Redis server port
        database_username (str): Redis username for authentication
        database_password (Optional[str]): Redis password for authentication
        logger (logging.Logger): Logger instance for database operations
        redis_client (Optional[Redis]): Redis client instance
    """

    def __init__(
        self,
        db_config: Optional[DatabaseConfig] = None,
    ):
        """
        Initialize the database manager with Redis connection parameters.

        Args:
            database_username (str): Redis username for authentication
            database_port (int): Redis server port number
            database_host (str, optional): Redis server hostname. Defaults to "localhost"
            database_password (Optional[str], optional): Redis password. Defaults to None
        """
        self.database_host = db_config.host
        self.database_port = db_config.port
        self.database_username = db_config.username
        self.database_password = db_config.password

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("ADAPTIQ-DatabaseManager")

        self.redis_client: Optional[Redis] = None

    def _connect(self) -> Redis:
        """Establish connection to Redis."""
        try:
            client = Redis(
                host=self.database_host,
                port=int(self.database_port),
                username=self.database_username,
                password=self.database_password,
                decode_responses=True,
                ssl=False
            )

            # Test connection
            client.ping()
            self.logger.info("Connected to Redis Cloud successfully!")
            return client

        except redis.ConnectionError as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise

    def save_q_table_to_database(self, q_table: QTablePayload) -> bool:
        """
        Saves QTablePayload to Redis Cloud using the version as the key name.

        Args:
            q_table: QTablePayload object to save

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Connect to Redis
            r = self._connect()

            # Save to Redis using version as key
            redis_key = f"adaptiq:q_table:{q_table.version}"

            # Serialize the Pydantic model to JSON
            json_data = q_table.model_dump_json(indent=2)

            r.set(redis_key, json_data)
            self.logger.info(f"Data saved to Redis with key: '{redis_key}'")

            # Verify the data was saved
            retrieved = r.get(redis_key)
            if retrieved:
                self.logger.info("Data verified in Redis!")
                self.logger.info(f"Data size: {len(retrieved)} characters")
                self.logger.info(f"Version: {q_table.version}")
                self.logger.info(f"Timestamp: {q_table.timestamp}")
                return True
            else:
                self.logger.warning("Data verification failed - unable to retrieve saved data")
                return False

        except redis.ConnectionError as e:
            self.logger.error(f"Connection error while saving to Redis: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to save to Redis: {e}")
            return False

    def retrieve_q_table_from_database(self, version: str) -> Optional[QTablePayload]:
        """
        Retrieves QTablePayload from Redis using the version key.

        Args:
            version: Version string to retrieve

        Returns:
            QTablePayload if found, None otherwise
        """
        try:
            # Connect to Redis
            r = self._connect()

            # Retrieve data using version as key
            redis_key = f"adaptiq:q_table:{version}"
            data_str = r.get(redis_key)

            if data_str:
                # Parse JSON and validate with Pydantic
                q_table = QTablePayload.model_validate_json(data_str)
                self.logger.info(f"Retrieved Q-Table for version: {version}")
                self.logger.info(f"Timestamp: {q_table.timestamp}")
                self.logger.info(f"Number of states: {len(q_table.Q_table)}")
                self.logger.info(f"Seen states: {len(q_table.seen_states)}")
                return q_table
            else:
                self.logger.warning(f"No data found in Redis for version: {version}")
                return None

        except redis.ConnectionError as e:
            self.logger.error(f"Connection error while retrieving from Redis: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve from Redis: {e}")
            return None

    def get_logger(self):
        """Get the logger instance."""
        return self.logger