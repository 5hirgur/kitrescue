#include <Arduino.h>
#include "esp_camera.h"
#include "board_config.h"

constexpr uint32_t SERIAL_BAUD = 921600;
esp_err_t camera_status = ESP_FAIL;
uint32_t sequence = 0;

// Standard CRC32, matching Python's zlib.crc32().
uint32_t crc32(const uint8_t *data, size_t length) {
  uint32_t crc = 0xFFFFFFFF;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1) ^ ((crc & 1) ? 0xEDB88320 : 0);
    }
  }
  return crc ^ 0xFFFFFFFF;
}

void initialise_camera() {
  camera_config_t c = {};

  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;

  c.pin_d0 = Y2_GPIO_NUM;
  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM;

  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;

  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_VGA;  // 640 × 480
  c.jpeg_quality = 12;          // Lower number means higher quality.
  c.fb_count = 1;
  c.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  c.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;

  camera_status = esp_camera_init(&c);
}

void send_frame() {
  camera_fb_t *frame = nullptr;

  if (camera_status == ESP_OK) {
    // Discard buffered frames after the capture request.
    for (int i = 0; i < 2; ++i) {
      camera_fb_t *discard = esp_camera_fb_get();
      if (discard) {
        esp_camera_fb_return(discard);
      }
    }
    frame = esp_camera_fb_get();
  }

  // Protocol: magic + sequence + JPEG length + JPEG CRC32 + JPEG.
  uint32_t length = frame ? static_cast<uint32_t>(frame->len) : 0;
  uint32_t checksum = frame ? crc32(frame->buf, frame->len) : 0;
  uint32_t header[3] = {++sequence, length, checksum};

  Serial.write(reinterpret_cast<const uint8_t *>("KRC1"), 4);
  Serial.write(reinterpret_cast<const uint8_t *>(header), sizeof(header));

  if (frame) {
    Serial.write(frame->buf, frame->len);
  }

  Serial.flush();

  if (frame) {
    esp_camera_fb_return(frame);
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  Serial.setDebugOutput(false);
  Serial.setTimeout(100);
  initialise_camera();
}

void loop() {
  if (!Serial.available()) {
    delay(2);
    return;
  }

  String command = Serial.readStringUntil('\n');
  command.trim();

  if (command == "PING") {
    if (camera_status == ESP_OK) {
      Serial.printf("KRCAM/1 READY 640 480 PSRAM=%d\n", psramFound());
    } else {
      Serial.printf("KRCAM/1 ERROR camera_init=0x%X\n", camera_status);
    }
  } else if (command == "CAPTURE") {
    send_frame();
  }
}