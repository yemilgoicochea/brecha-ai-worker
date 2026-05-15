"""Google Cloud Pub/Sub service."""

import json
import logging
from concurrent.futures import TimeoutError as ConcurrentTimeoutError
from typing import Any, Callable, Dict, Optional

from google.cloud import pubsub_v1

from app.core.config import settings

logger = logging.getLogger(__name__)


class PubSubService:
    """Service for managing Google Cloud Pub/Sub interactions."""

    def __init__(self):
        """Initialize the Pub/Sub service."""
        try:
            self.project_id = settings.GCP_PROJECT_ID
            self.subscription_id = settings.PUBSUB_SUBSCRIPTION_ID
            self.topic_id = settings.PUBSUB_TOPIC_ID

            # Create subscriber client
            self.subscriber_client = pubsub_v1.SubscriberClient()
            self.subscription_path = self.subscriber_client.subscription_path(
                self.project_id, self.subscription_id
            )

            logger.info(
                f"Pub/Sub service initialized. "
                f"Project: {self.project_id}, "
                f"Subscription: {self.subscription_id}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Pub/Sub service: {str(e)}")
            raise

    def listen(self, callback: Callable[[Dict[str, Any]], bool], timeout: int = 300):
        """
        Listen to messages from the subscription.

        Args:
            callback: Function to call for each message. Should return True if message was processed successfully.
            timeout: Timeout in seconds

        Raises:
            Exception: If listening fails
        """

        def message_handler(message):
            """Handle incoming message."""
            try:
                # Parse message data
                payload = json.loads(message.data.decode("utf-8"))
                logger.info(f"Received message: {payload.get('query_id', 'unknown')}")

                # Call the callback function
                success = callback(payload)

                # Acknowledge the message only if processing was successful
                if success:
                    message.ack()
                    logger.info(f"Message acknowledged: {payload.get('query_id', 'unknown')}")
                else:
                    # Don't acknowledge, let it retry
                    logger.warning(f"Message processing failed, will retry: {payload.get('query_id', 'unknown')}")

            except Exception as e:
                logger.error(f"Error processing message: {str(e)}", exc_info=True)
                # Don't acknowledge on error, let it retry

        logger.info(f"Starting to listen on {self.subscription_path}")

        # Set up the streaming pull
        streaming_pull_future = self.subscriber_client.subscribe(
            self.subscription_path,
            callback=message_handler,
            flow_control=pubsub_v1.types.FlowControl(
                max_messages=settings.PUBSUB_MAX_MESSAGES,
                max_bytes=100 * 1024 * 1024,  # 100 MB
            ),
        )

        # Wrap to handle exceptions
        with self.subscriber_client:
            try:
                while True:
                    try:
                        streaming_pull_future.result(timeout=1)
                        break
                    except ConcurrentTimeoutError:
                        pass  # sigue escuchando
            except KeyboardInterrupt:
                logger.info("Shutting down worker...")
                streaming_pull_future.cancel()
            except Exception as e:
                logger.error(f"Error in message listener: {str(e)}", exc_info=True)
                streaming_pull_future.cancel()
                raise

    def publish_status_update(self, message: Dict[str, Any]) -> str:
        """
        Publish a status update message.

        Args:
            message: Message data to publish

        Returns:
            Message ID
        """
        try:
            publisher_client = pubsub_v1.PublisherClient()
            topic_path = publisher_client.topic_path(self.project_id, self.topic_id)

            # Publish message
            message_json = json.dumps(message)
            future = publisher_client.publish(topic_path, message_json.encode("utf-8"))
            message_id = future.result()

            logger.info(f"Published status update: {message_id}")
            return message_id

        except Exception as e:
            logger.error(f"Failed to publish status update: {str(e)}")
            raise
