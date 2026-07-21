/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Jumping_Sensors — ISM330DHCX IMU over SPI + FDCAN send
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "ism330dhcx_reg.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct {
  float position_target;
  float velocity_target;
  float kp;
  float kd;
  float torque_ff;
} CAN_Command_t;

typedef struct {
  float position_actual;
  float velocity_actual;
  float torque_actual;
  uint8_t motor_id;
  uint8_t state_nibble;
  uint8_t error_code;
} CAN_MotorFeedback_t;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#define CS_IMU_ASSERT()    HAL_GPIO_WritePin(CS_ACCEL_GPIO_Port, CS_ACCEL_Pin, GPIO_PIN_RESET)
#define CS_IMU_DEASSERT()  HAL_GPIO_WritePin(CS_ACCEL_GPIO_Port, CS_ACCEL_Pin, GPIO_PIN_SET)

/* CAN wire format shared with motor_control_gui.py. */
#define SENSOR_ID                0x04U
#define SENSOR_CTRL_ID           (SENSOR_ID + 0x200U)
#define SENSOR_FEEDBACK_ID       (SENSOR_ID + 0x100U)
#define SENSOR_AUX_FEEDBACK_ID   (SENSOR_ID + 0x101U)
#define SENSOR_ACCEL_FEEDBACK_ID (SENSOR_ID + 0x102U)
#define SENSOR_YAW_FEEDBACK_ID   (SENSOR_ID + 0x103U)
#define FLYWHEEL_X_ID            0x01U
#define FLYWHEEL_Y_ID            0x02U
#define FLYWHEEL_X_CTRL_ID       (FLYWHEEL_X_ID + 0x200U)
#define FLYWHEEL_Y_CTRL_ID       (FLYWHEEL_Y_ID + 0x200U)
#define FLYWHEEL_X_FEEDBACK_ID   (FLYWHEEL_X_ID + 0x100U)
#define FLYWHEEL_Y_FEEDBACK_ID   (FLYWHEEL_Y_ID + 0x100U)
#define CMD_MOTOR_START          0x01U
#define CMD_MOTOR_STOP           0x02U
#define CMD_MOTOR_ALIGN          0x03U
#define SENSOR_STATE_RUN         0x01U
#define SENSOR_ERROR_IMU_DATA    0x08U
#define SENSOR_ERROR_TOF_DATA    0x10U
#define SENSOR_ERROR_TOF_INIT    0x20U
#define SENSOR_ERROR_CAN_TX      0x40U
#define SENSOR_ERROR_IMU_ID      0x80U
#define CAN_TX_PERIOD_MS         20U
#define CAN_CTRL_PERIOD_MS       1U
#define CAN_HEALTH_PERIOD_MS     10U
#define MOTOR_CMD_OVERRIDE_MS    250U
#define MOTOR_ALIGN_OVERRIDE_MS 2000U
#define SENSOR_LED_CMD_MAGIC     0xA5U
#define SENSOR_LED_COUNT         4U
#define SENSOR_LED_TEST_CYCLE_ENABLE 1U
#define SENSOR_LED_TEST_PERIOD_MS   250U
#define UART_DEBUG_ENABLE             0U
#define UART_PERIODIC_DEBUG_ENABLE    0U
#define WS2812_BITS_PER_LED      24U
#define WS2812_RESET_SLOTS       64U
#define WS2812_BITRATE_HZ        800000U

#define CAN_ANGLE_MIN   -3.14159265f
#define CAN_ANGLE_MAX    3.14159265f
#define CAN_RATE_MIN   -1500.0f
#define CAN_RATE_MAX    1500.0f
#define CAN_CMD_POS_MIN      -12.5f
#define CAN_CMD_POS_MAX       12.5f
#define CAN_CMD_VEL_MIN      -45.0f
#define CAN_CMD_VEL_MAX       45.0f
#define CAN_CMD_KP_MIN         0.0f
#define CAN_CMD_KP_MAX       500.0f
#define CAN_CMD_KD_MIN         0.0f
#define CAN_CMD_KD_MAX        15.0f
#define CAN_CMD_TORQUE_MIN    -1.0f
#define CAN_CMD_TORQUE_MAX     1.0f
#define CAN_FB_POS_MIN       -12.5f
#define CAN_FB_POS_MAX        12.5f
#define CAN_FB_VEL_MIN     -1500.0f
#define CAN_FB_VEL_MAX      1500.0f
#define CAN_FB_TRQ_MIN        -1.0f
#define CAN_FB_TRQ_MAX         1.0f

#define LQR_X_K1             50.0f
#define LQR_X_K2             0.0f
#define LQR_X_K3             -0.01f
#define LQR_Y_K1             50.0f
#define LQR_Y_K2             0.0f
#define LQR_Y_K3             -0.01f
#define LQR_TORQUE_LIMIT      0.4f
#define LQR_THETA_SETPOINT_X  0.0f
#define LQR_THETA_SETPOINT_Y  0.0f
#define ACCEL_RAW_MAG_LO          0.92f
#define ACCEL_RAW_MAG_HI          1.08f
#define ACCEL_FILTERED_MAG_LO     0.98f
#define ACCEL_FILTERED_MAG_HI     1.02f
#define ACCEL_LPF_TAU_S           0.12f
#define ACCEL_RATE_LIMIT_RAD_S    0.75f
#define ACCEL_VALID_SAMPLES       40U
#define MADGWICK_BETA              0.25f
#define GYRO_MAX_BODY_RATE_RAD_S  12.0f
#define GYRO_MAX_SLEW_RAD_S2     400.0f
#define GYRO_SLEW_MARGIN_RAD_S     0.05f
#define GYRO_RESEED_SAMPLES         3U
#define GYRO_RECOVERY_SAMPLES      20U

#define VL53L4CD_I2C_ADDR                 (0x29U << 1)
#define VL53L4CD_REG_VHV_CONFIG_TIMEOUT   0x0008U
#define VL53L4CD_REG_GPIO_HV_MUX_CTRL     0x0030U
#define VL53L4CD_REG_GPIO_TIO_HV_STATUS   0x0031U
#define VL53L4CD_REG_SYSTEM_INTERRUPT_CLR 0x0086U
#define VL53L4CD_REG_SYSTEM_START         0x0087U
#define VL53L4CD_REG_RESULT_DISTANCE      0x0096U
#define VL53L4CD_REG_FIRMWARE_STATUS      0x00E5U
#define VL53L4CD_REG_MODEL_ID             0x010FU
#define VL53L4CD_BOOT_OK                  0x03U
#define VL53L4CD_MODEL_ID_MSB             0xEBU
#define VL53L4CD_MODEL_ID_LSB             0xAAU
#define VL53L4CD_INIT_TIMEOUT_MS          1000U

static const uint8_t vl53l4cd_default_config[] = {
  0x12U, 0x00U, 0x00U, 0x11U, 0x02U, 0x00U, 0x02U, 0x08U, 0x00U, 0x08U, 0x10U,
  0x01U, 0x01U, 0x00U, 0x00U, 0x00U, 0x00U, 0xFFU, 0x00U, 0x0FU, 0x00U, 0x00U,
  0x00U, 0x00U, 0x00U, 0x20U, 0x0BU, 0x00U, 0x00U, 0x02U, 0x14U, 0x21U, 0x00U,
  0x00U, 0x05U, 0x00U, 0x00U, 0x00U, 0x00U, 0xC8U, 0x00U, 0x00U, 0x38U, 0xFFU,
  0x01U, 0x00U, 0x08U, 0x00U, 0x00U, 0x01U, 0xCCU, 0x07U, 0x01U, 0xF1U, 0x05U,
  0x00U, 0xA0U, 0x00U, 0x80U, 0x08U, 0x38U, 0x00U, 0x00U, 0x00U, 0x00U, 0x0FU,
  0x89U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, 0x01U, 0x07U, 0x05U,
  0x06U, 0x06U, 0x00U, 0x00U, 0x02U, 0xC7U, 0xFFU, 0x9BU, 0x00U, 0x00U, 0x00U,
  0x01U, 0x00U, 0x00U
};

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
FDCAN_HandleTypeDef hfdcan1;

I2C_HandleTypeDef hi2c1;

UART_HandleTypeDef hlpuart1;
UART_HandleTypeDef huart1;

SPI_HandleTypeDef hspi1;

TIM_HandleTypeDef htim2;

/* USER CODE BEGIN PV */
static stmdev_ctx_t imu_ctx;
static char uart_buf[128];

static float    theta_x       = 0.0f;
static float    theta_y       = 0.0f;
static float    theta_z       = 0.0f;
static float    attitude_q_w  = 1.0f;
static float    attitude_q_x  = 0.0f;
static float    attitude_q_y  = 0.0f;
static float    attitude_q_z  = 0.0f;
static float    accel_lpf_x_g = 0.0f;
static float    accel_lpf_y_g = 0.0f;
static float    accel_lpf_z_g = 1.0f;
static float    accel_angle_x = 0.0f;
static float    accel_angle_y = 0.0f;
static uint16_t accel_valid_count = 0U;
static uint8_t  accel_angles_valid = 0U;
static float    gyro_history_x[2] = {0.0f, 0.0f};
static float    gyro_history_y[2] = {0.0f, 0.0f};
static float    gyro_history_z[2] = {0.0f, 0.0f};
static float    gyro_accepted_x = 0.0f;
static float    gyro_accepted_y = 0.0f;
static float    gyro_accepted_z = 0.0f;
static float    gyro_reseed_x = 0.0f;
static float    gyro_reseed_y = 0.0f;
static float    gyro_reseed_z = 0.0f;
static uint8_t  gyro_history_count = 0U;
static uint8_t  gyro_reseed_count = 0U;
static uint8_t  gyro_reject_latched = 0U;
static uint8_t  gyro_recovery_count = 0U;
static uint8_t  imu_config_valid = 0U;
static volatile uint16_t tof_distance_mm = 0U;
static uint8_t  sensor_error_code = SENSOR_ERROR_IMU_ID;
static uint32_t ctrl_t_prev   = 0;
static uint32_t can_tx_prev   = 0;
static uint32_t can_ctrl_prev = 0;
static uint32_t can_health_prev = 0;
static volatile uint32_t can_rx_count  = 0;
static HAL_StatusTypeDef can_last_tx_status = HAL_OK;
static HAL_StatusTypeDef can_last_ctrl_status = HAL_OK;
static volatile CAN_MotorFeedback_t flywheel_x_feedback = {0};
static volatile CAN_MotorFeedback_t flywheel_y_feedback = {0};
static volatile uint32_t flywheel_x_override_until_ms = 0U;
static volatile uint32_t flywheel_y_override_until_ms = 0U;
static volatile uint8_t flywheel_x_enabled = 0U;
static volatile uint8_t flywheel_y_enabled = 0U;
#if (UART_PERIODIC_DEBUG_ENABLE == 1U)
static uint32_t print_counter = 0;
static uint32_t loop_count    = 0;
static uint32_t print_t_prev  = 0;
#endif
static volatile uint8_t led_r_cmd = 0U;
static volatile uint8_t led_g_cmd = 0U;
static volatile uint8_t led_b_cmd = 0U;
static volatile uint8_t led_update_pending = 1U;
static uint32_t led_test_prev_ms = 0U;
static uint8_t led_test_idx = 0U;
DMA_HandleTypeDef hdma_tim2_ch2;
static uint16_t led_pwm_buf[(SENSOR_LED_COUNT * WS2812_BITS_PER_LED) + WS2812_RESET_SLOTS];
static uint16_t led_pwm_duty_0 = 0U;
static uint16_t led_pwm_duty_1 = 0U;
static volatile uint8_t led_dma_done = 1U;
static uint8_t led_dma_initialized = 0U;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
static void MX_LPUART1_UART_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_SPI1_Init(void);
static void MX_TIM2_Init(void);
static void MX_FDCAN1_Init(void);
/* USER CODE BEGIN PFP */
static int32_t imu_spi_write(void *handle, uint8_t reg, const uint8_t *buf, uint16_t len);
static int32_t imu_spi_read(void *handle, uint8_t reg, uint8_t *buf, uint16_t len);
static void    uart_print(const char *str);
static uint16_t can_map_float_to_u16(float x, float x_min, float x_max);
static int      can_map_float_to_int(float x, float x_min, float x_max, int bits);
static float    can_map_u16_to_float(uint16_t x, float x_min, float x_max);
static float    wrap_to_pi(float angle);
static float    median_of_three(float a, float b, float c);
static uint8_t  gyro_filter_update(float raw_x, float raw_y, float raw_z,
                                   float dt, uint8_t sample_valid,
                                   float *filtered_x, float *filtered_y,
                                   float *filtered_z);
static void     madgwick_set_tilt(float roll, float pitch);
static void     madgwick_update_imu(float gx, float gy, float gz,
                                    float ax, float ay, float az,
                                    float dt, uint8_t use_accel);
static void     madgwick_get_attitude(float *roll, float *pitch, float *yaw);
static void     can_pack_command(uint8_t *data, const CAN_Command_t *cmd);
static void     can_unpack_motor_feedback(const uint8_t *data, CAN_MotorFeedback_t *fb);
static HAL_StatusTypeDef can_send_motor_command(uint16_t motor_id, const CAN_Command_t *cmd);
static HAL_StatusTypeDef tof_write_bytes(uint16_t reg, const uint8_t *data, uint16_t len);
static HAL_StatusTypeDef tof_read_bytes(uint16_t reg, uint8_t *data, uint16_t len);
static HAL_StatusTypeDef tof_write_u8(uint16_t reg, uint8_t value);
static HAL_StatusTypeDef tof_read_u8(uint16_t reg, uint8_t *value);
static HAL_StatusTypeDef tof_read_u16(uint16_t reg, uint16_t *value);
static HAL_StatusTypeDef tof_is_data_ready(uint8_t *ready);
static HAL_StatusTypeDef tof_init(void);
static HAL_StatusTypeDef tof_try_read_distance(uint16_t *distance_mm);
static HAL_StatusTypeDef can_send_sensor_feedback(float body_angle,
                                                   float gyro_rate,
                                                   uint16_t tof_distance_mm);
static HAL_StatusTypeDef can_send_sensor_aux_feedback(float body_angle_y,
                                                      float gyro_rate_y);
static HAL_StatusTypeDef can_send_sensor_accel_feedback(float accel_angle_x_value,
                                                        float accel_angle_y_value);
static HAL_StatusTypeDef can_send_sensor_yaw_feedback(float body_angle_z,
                                                       float gyro_rate_z);
static void sensor_led_init(void);
static void sensor_led_set_all(uint8_t r, uint8_t g, uint8_t b);
static void sensor_led_encode_byte(uint8_t value, uint32_t *index);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

static void uart_print(const char *str)
{
#if (UART_DEBUG_ENABLE == 1U)
  HAL_UART_Transmit(&hlpuart1, (uint8_t *)str, strlen(str), 100);
#else
  (void)str;
#endif
}

/* ── ISM330DHCX SPI callbacks ───────────────────────────────────────────── */
static int32_t imu_spi_write(void *handle, uint8_t reg, const uint8_t *buf, uint16_t len)
{
  HAL_StatusTypeDef status;

  CS_IMU_ASSERT();
  status = HAL_SPI_Transmit((SPI_HandleTypeDef *)handle, &reg, 1, 10);
  if (status == HAL_OK)
  {
    status = HAL_SPI_Transmit((SPI_HandleTypeDef *)handle, (uint8_t *)buf, len, 10);
  }
  CS_IMU_DEASSERT();
  return (status == HAL_OK) ? 0 : -1;
}

static int32_t imu_spi_read(void *handle, uint8_t reg, uint8_t *buf, uint16_t len)
{
  HAL_StatusTypeDef status;

  reg |= 0x80U;
  CS_IMU_ASSERT();
  status = HAL_SPI_Transmit((SPI_HandleTypeDef *)handle, &reg, 1, 10);
  if (status == HAL_OK)
  {
    status = HAL_SPI_Receive((SPI_HandleTypeDef *)handle, buf, len, 10);
  }
  CS_IMU_DEASSERT();
  return (status == HAL_OK) ? 0 : -1;
}

/* ── FDCAN helpers ──────────────────────────────────────────────────────── */
static uint16_t can_map_float_to_u16(float x, float x_min, float x_max)
{
  if (x < x_min) x = x_min;
  if (x > x_max) x = x_max;
  return (uint16_t)((x - x_min) * 65535.0f / (x_max - x_min));
}

static int can_map_float_to_int(float x, float x_min, float x_max, int bits)
{
  if (x < x_min) x = x_min;
  if (x > x_max) x = x_max;
  return (int)((x - x_min) * (float)((1 << bits) - 1) / (x_max - x_min));
}

static float can_map_u16_to_float(uint16_t x, float x_min, float x_max)
{
  return x_min + ((float)x * (x_max - x_min) / 65535.0f);
}

static float wrap_to_pi(float angle)
{
  while (angle > 3.14159265f)
  {
    angle -= 2.0f * 3.14159265f;
  }
  while (angle < -3.14159265f)
  {
    angle += 2.0f * 3.14159265f;
  }
  return angle;
}

static float median_of_three(float a, float b, float c)
{
  if (a > b)
  {
    float tmp = a;
    a = b;
    b = tmp;
  }
  if (b > c)
  {
    b = c;
  }
  return (a > b) ? a : b;
}

static uint8_t gyro_filter_update(float raw_x, float raw_y, float raw_z,
                                  float dt, uint8_t sample_valid,
                                  float *filtered_x, float *filtered_y,
                                  float *filtered_z)
{
  float gate_dt = (dt >= 0.0005f) ? dt : 0.001f;
  float max_delta = GYRO_MAX_SLEW_RAD_S2 * gate_dt + GYRO_SLEW_MARGIN_RAD_S;

  if (sample_valid == 0U ||
      fabsf(raw_x) > GYRO_MAX_BODY_RATE_RAD_S ||
      fabsf(raw_y) > GYRO_MAX_BODY_RATE_RAD_S ||
      fabsf(raw_z) > GYRO_MAX_BODY_RATE_RAD_S)
  {
    *filtered_x = 0.0f;
    *filtered_y = 0.0f;
    *filtered_z = 0.0f;
    gyro_reseed_count = 0U;
    gyro_reject_latched = 1U;
    return 0U;
  }

  if (gyro_history_count != 0U &&
      (fabsf(raw_x - gyro_history_x[gyro_history_count - 1U]) > max_delta ||
       fabsf(raw_y - gyro_history_y[gyro_history_count - 1U]) > max_delta ||
       fabsf(raw_z - gyro_history_z[gyro_history_count - 1U]) > max_delta))
  {
    if (gyro_reseed_count == 0U ||
        fabsf(raw_x - gyro_reseed_x) > max_delta ||
        fabsf(raw_y - gyro_reseed_y) > max_delta ||
        fabsf(raw_z - gyro_reseed_z) > max_delta)
    {
      gyro_reseed_count = 1U;
    }
    else if (gyro_reseed_count < GYRO_RESEED_SAMPLES)
    {
      gyro_reseed_count++;
    }

    gyro_reseed_x = raw_x;
    gyro_reseed_y = raw_y;
    gyro_reseed_z = raw_z;
    gyro_reject_latched = 1U;

    if (gyro_reseed_count < GYRO_RESEED_SAMPLES)
    {
      *filtered_x = 0.0f;
      *filtered_y = 0.0f;
      *filtered_z = 0.0f;
      return 0U;
    }

    gyro_history_count = 0U;
    gyro_reseed_count = 0U;
  }
  else
  {
    gyro_reseed_count = 0U;
  }

  if (gyro_history_count < 2U)
  {
    gyro_history_x[gyro_history_count] = raw_x;
    gyro_history_y[gyro_history_count] = raw_y;
    gyro_history_z[gyro_history_count] = raw_z;
    gyro_history_count++;
    gyro_accepted_x = raw_x;
    gyro_accepted_y = raw_y;
    gyro_accepted_z = raw_z;
  }
  else
  {
    gyro_accepted_x = median_of_three(gyro_history_x[0], gyro_history_x[1], raw_x);
    gyro_accepted_y = median_of_three(gyro_history_y[0], gyro_history_y[1], raw_y);
    gyro_accepted_z = median_of_three(gyro_history_z[0], gyro_history_z[1], raw_z);
    gyro_history_x[0] = gyro_history_x[1];
    gyro_history_y[0] = gyro_history_y[1];
    gyro_history_z[0] = gyro_history_z[1];
    gyro_history_x[1] = raw_x;
    gyro_history_y[1] = raw_y;
    gyro_history_z[1] = raw_z;
  }

  *filtered_x = gyro_accepted_x;
  *filtered_y = gyro_accepted_y;
  *filtered_z = gyro_accepted_z;
  return 1U;
}

static void madgwick_set_tilt(float roll, float pitch)
{
  float yaw = atan2f(2.0f * (attitude_q_w * attitude_q_z +
                             attitude_q_x * attitude_q_y),
                     1.0f - 2.0f * (attitude_q_y * attitude_q_y +
                                    attitude_q_z * attitude_q_z));
  float half_roll = 0.5f * roll;
  float half_pitch = 0.5f * pitch;
  float half_yaw = 0.5f * yaw;
  float cr = cosf(half_roll);
  float sr = sinf(half_roll);
  float cp = cosf(half_pitch);
  float sp = sinf(half_pitch);
  float cy = cosf(half_yaw);
  float sy = sinf(half_yaw);

  attitude_q_w = cr * cp * cy + sr * sp * sy;
  attitude_q_x = sr * cp * cy - cr * sp * sy;
  attitude_q_y = cr * sp * cy + sr * cp * sy;
  attitude_q_z = cr * cp * sy - sr * sp * cy;
}

static void madgwick_update_imu(float gx, float gy, float gz,
                                float ax, float ay, float az,
                                float dt, uint8_t use_accel)
{
  float qw = attitude_q_w;
  float qx = attitude_q_x;
  float qy = attitude_q_y;
  float qz = attitude_q_z;
  float q_dot_w = 0.5f * (-qx * gx - qy * gy - qz * gz);
  float q_dot_x = 0.5f * ( qw * gx + qy * gz - qz * gy);
  float q_dot_y = 0.5f * ( qw * gy - qx * gz + qz * gx);
  float q_dot_z = 0.5f * ( qw * gz + qx * gy - qy * gx);

  if (use_accel != 0U)
  {
    float accel_norm_sq = ax * ax + ay * ay + az * az;
    if (accel_norm_sq > 0.0f)
    {
      float inv_accel_norm = 1.0f / sqrtf(accel_norm_sq);
      float s_w;
      float s_x;
      float s_y;
      float s_z;
      float step_norm_sq;
      float inv_step_norm;
      float two_qw;
      float two_qx;
      float two_qy;
      float two_qz;
      float four_qw;
      float four_qx;
      float four_qy;
      float eight_qx;
      float eight_qy;
      float qw_qw;
      float qx_qx;
      float qy_qy;
      float qz_qz;

      ax *= inv_accel_norm;
      ay *= inv_accel_norm;
      az *= inv_accel_norm;

      two_qw = 2.0f * qw;
      two_qx = 2.0f * qx;
      two_qy = 2.0f * qy;
      two_qz = 2.0f * qz;
      four_qw = 4.0f * qw;
      four_qx = 4.0f * qx;
      four_qy = 4.0f * qy;
      eight_qx = 8.0f * qx;
      eight_qy = 8.0f * qy;
      qw_qw = qw * qw;
      qx_qx = qx * qx;
      qy_qy = qy * qy;
      qz_qz = qz * qz;

      s_w = four_qw * qy_qy + two_qy * ax +
            four_qw * qx_qx - two_qx * ay;
      s_x = four_qx * qz_qz - two_qz * ax +
            4.0f * qw_qw * qx - two_qw * ay -
            four_qx + eight_qx * qx_qx +
            eight_qx * qy_qy + four_qx * az;
      s_y = 4.0f * qw_qw * qy + two_qw * ax +
            four_qy * qz_qz - two_qz * ay -
            four_qy + eight_qy * qx_qx +
            eight_qy * qy_qy + four_qy * az;
      s_z = 4.0f * qx_qx * qz - two_qx * ax +
            4.0f * qy_qy * qz - two_qy * ay;

      step_norm_sq = s_w * s_w + s_x * s_x + s_y * s_y + s_z * s_z;
      if (step_norm_sq > 0.0f)
      {
        inv_step_norm = 1.0f / sqrtf(step_norm_sq);
        q_dot_w -= MADGWICK_BETA * s_w * inv_step_norm;
        q_dot_x -= MADGWICK_BETA * s_x * inv_step_norm;
        q_dot_y -= MADGWICK_BETA * s_y * inv_step_norm;
        q_dot_z -= MADGWICK_BETA * s_z * inv_step_norm;
      }
    }
  }

  if (dt > 0.0f)
  {
    float quaternion_norm_sq;
    float inv_quaternion_norm;

    qw += q_dot_w * dt;
    qx += q_dot_x * dt;
    qy += q_dot_y * dt;
    qz += q_dot_z * dt;
    quaternion_norm_sq = qw * qw + qx * qx + qy * qy + qz * qz;
    if (quaternion_norm_sq > 0.0f)
    {
      inv_quaternion_norm = 1.0f / sqrtf(quaternion_norm_sq);
      attitude_q_w = qw * inv_quaternion_norm;
      attitude_q_x = qx * inv_quaternion_norm;
      attitude_q_y = qy * inv_quaternion_norm;
      attitude_q_z = qz * inv_quaternion_norm;
    }
  }
}

static void madgwick_get_attitude(float *roll, float *pitch, float *yaw)
{
  float sin_pitch = 2.0f * (attitude_q_w * attitude_q_y -
                            attitude_q_z * attitude_q_x);
  if (sin_pitch > 1.0f)
  {
    sin_pitch = 1.0f;
  }
  else if (sin_pitch < -1.0f)
  {
    sin_pitch = -1.0f;
  }

  *roll = atan2f(2.0f * (attitude_q_w * attitude_q_x +
                         attitude_q_y * attitude_q_z),
                 1.0f - 2.0f * (attitude_q_x * attitude_q_x +
                                 attitude_q_y * attitude_q_y));
  *pitch = asinf(sin_pitch);
  *yaw = atan2f(2.0f * (attitude_q_w * attitude_q_z +
                        attitude_q_x * attitude_q_y),
                1.0f - 2.0f * (attitude_q_y * attitude_q_y +
                                attitude_q_z * attitude_q_z));
}

static void can_pack_command(uint8_t *data, const CAN_Command_t *cmd)
{
  uint16_t p_int = (uint16_t)can_map_float_to_int(cmd->position_target, CAN_CMD_POS_MIN, CAN_CMD_POS_MAX, 16);
  uint16_t v_int = (uint16_t)can_map_float_to_int(cmd->velocity_target, CAN_CMD_VEL_MIN, CAN_CMD_VEL_MAX, 12);
  uint16_t kp_int = (uint16_t)can_map_float_to_int(cmd->kp, CAN_CMD_KP_MIN, CAN_CMD_KP_MAX, 12);
  uint16_t kd_int = (uint16_t)can_map_float_to_int(cmd->kd, CAN_CMD_KD_MIN, CAN_CMD_KD_MAX, 12);
  uint16_t t_int = (uint16_t)can_map_float_to_int(cmd->torque_ff, CAN_CMD_TORQUE_MIN, CAN_CMD_TORQUE_MAX, 12);

  data[0] = (uint8_t)((p_int >> 8) & 0xFFU);
  data[1] = (uint8_t)(p_int & 0xFFU);
  data[2] = (uint8_t)((v_int >> 4) & 0xFFU);
  data[3] = (uint8_t)(((v_int & 0x0FU) << 4) | ((kp_int >> 8) & 0x0FU));
  data[4] = (uint8_t)(kp_int & 0xFFU);
  data[5] = (uint8_t)((kd_int >> 4) & 0xFFU);
  data[6] = (uint8_t)(((kd_int & 0x0FU) << 4) | ((t_int >> 8) & 0x0FU));
  data[7] = (uint8_t)(t_int & 0xFFU);
}

static void can_unpack_motor_feedback(const uint8_t *data, CAN_MotorFeedback_t *fb)
{
  uint16_t p_int = ((uint16_t)data[2] << 8) | data[3];
  uint16_t v_int = ((uint16_t)data[4] << 8) | data[5];
  uint16_t t_int = ((uint16_t)data[6] << 8) | data[7];
  fb->state_nibble = (data[0] >> 4) & 0x0FU;
  fb->motor_id = data[0] & 0x0FU;
  fb->error_code = data[1];
  fb->position_actual = can_map_u16_to_float(p_int, CAN_FB_POS_MIN, CAN_FB_POS_MAX);
  fb->velocity_actual = can_map_u16_to_float(v_int, CAN_FB_VEL_MIN, CAN_FB_VEL_MAX);
  fb->torque_actual = can_map_u16_to_float(t_int, CAN_FB_TRQ_MIN, CAN_FB_TRQ_MAX);
}

static HAL_StatusTypeDef can_send_motor_command(uint16_t motor_id, const CAN_Command_t *cmd)
{
  FDCAN_TxHeaderTypeDef tx_hdr = {0};
  uint8_t tx_data[8];
  can_pack_command(tx_data, cmd);
  tx_hdr.Identifier          = motor_id;
  tx_hdr.IdType              = FDCAN_STANDARD_ID;
  tx_hdr.TxFrameType         = FDCAN_DATA_FRAME;
  tx_hdr.DataLength          = FDCAN_DLC_BYTES_8;
  tx_hdr.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
  tx_hdr.BitRateSwitch       = FDCAN_BRS_OFF;
  tx_hdr.FDFormat            = FDCAN_CLASSIC_CAN;
  tx_hdr.TxEventFifoControl  = FDCAN_NO_TX_EVENTS;
  tx_hdr.MessageMarker       = 0;
  return HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan1, &tx_hdr, tx_data);
}

static HAL_StatusTypeDef tof_write_bytes(uint16_t reg, const uint8_t *data, uint16_t len)
{
  return HAL_I2C_Mem_Write(&hi2c1,
                           VL53L4CD_I2C_ADDR,
                           reg,
                           I2C_MEMADD_SIZE_16BIT,
                           (uint8_t *)data,
                           len,
                           100U);
}

static HAL_StatusTypeDef tof_read_bytes(uint16_t reg, uint8_t *data, uint16_t len)
{
  return HAL_I2C_Mem_Read(&hi2c1,
                          VL53L4CD_I2C_ADDR,
                          reg,
                          I2C_MEMADD_SIZE_16BIT,
                          data,
                          len,
                          100U);
}

static HAL_StatusTypeDef tof_write_u8(uint16_t reg, uint8_t value)
{
  return tof_write_bytes(reg, &value, 1U);
}

static HAL_StatusTypeDef tof_read_u8(uint16_t reg, uint8_t *value)
{
  return tof_read_bytes(reg, value, 1U);
}

static HAL_StatusTypeDef tof_read_u16(uint16_t reg, uint16_t *value)
{
  uint8_t raw[2];
  HAL_StatusTypeDef st = tof_read_bytes(reg, raw, 2U);
  if (st != HAL_OK)
  {
    return st;
  }
  *value = ((uint16_t)raw[0] << 8) | raw[1];
  return HAL_OK;
}

static HAL_StatusTypeDef tof_is_data_ready(uint8_t *ready)
{
  uint8_t gpio_mux = 0U;
  uint8_t gpio_status = 0U;
  HAL_StatusTypeDef st = tof_read_u8(VL53L4CD_REG_GPIO_HV_MUX_CTRL, &gpio_mux);
  if (st != HAL_OK)
  {
    return st;
  }
  st = tof_read_u8(VL53L4CD_REG_GPIO_TIO_HV_STATUS, &gpio_status);
  if (st != HAL_OK)
  {
    return st;
  }
  *ready = (uint8_t)((gpio_status & 0x01U) == (((gpio_mux & 0x10U) != 0U) ? 0U : 1U));
  return HAL_OK;
}

static HAL_StatusTypeDef tof_init(void)
{
  uint8_t id[2];
  uint8_t fw_status = 0U;
  uint8_t ready = 0U;
  uint32_t t0;
  HAL_StatusTypeDef st = tof_read_bytes(VL53L4CD_REG_MODEL_ID, id, 2U);
  if (st != HAL_OK)
  {
    return st;
  }
  if ((id[0] != VL53L4CD_MODEL_ID_MSB) || (id[1] != VL53L4CD_MODEL_ID_LSB))
  {
    return HAL_ERROR;
  }

  t0 = HAL_GetTick();
  do
  {
    st = tof_read_u8(VL53L4CD_REG_FIRMWARE_STATUS, &fw_status);
    if (st != HAL_OK)
    {
      return st;
    }
    if (fw_status == VL53L4CD_BOOT_OK)
    {
      break;
    }
    HAL_Delay(1U);
  } while ((HAL_GetTick() - t0) < VL53L4CD_INIT_TIMEOUT_MS);
  if (fw_status != VL53L4CD_BOOT_OK)
  {
    return HAL_TIMEOUT;
  }

  st = tof_write_bytes(0x002DU, vl53l4cd_default_config, (uint16_t)sizeof(vl53l4cd_default_config));
  if (st != HAL_OK)
  {
    return st;
  }

  st = tof_write_u8(VL53L4CD_REG_SYSTEM_START, 0x40U);
  if (st != HAL_OK)
  {
    return st;
  }

  t0 = HAL_GetTick();
  while ((HAL_GetTick() - t0) < VL53L4CD_INIT_TIMEOUT_MS)
  {
    st = tof_is_data_ready(&ready);
    if (st != HAL_OK)
    {
      return st;
    }
    if (ready != 0U)
    {
      break;
    }
    HAL_Delay(1U);
  }
  if (ready == 0U)
  {
    return HAL_TIMEOUT;
  }

  st = tof_write_u8(VL53L4CD_REG_SYSTEM_INTERRUPT_CLR, 0x01U);
  if (st != HAL_OK)
  {
    return st;
  }
  st = tof_write_u8(VL53L4CD_REG_SYSTEM_START, 0x00U);
  if (st != HAL_OK)
  {
    return st;
  }

  st = tof_write_u8(VL53L4CD_REG_VHV_CONFIG_TIMEOUT, 0x09U);
  if (st != HAL_OK)
  {
    return st;
  }
  st = tof_write_u8(0x000BU, 0x00U);
  if (st != HAL_OK)
  {
    return st;
  }
  {
    uint8_t reg24_cfg[2] = {0x05U, 0x00U};
    st = tof_write_bytes(0x0024U, reg24_cfg, 2U);
    if (st != HAL_OK)
    {
      return st;
    }
  }
  st = tof_write_u8(VL53L4CD_REG_SYSTEM_START, 0x21U);
  return st;
}

static HAL_StatusTypeDef tof_try_read_distance(uint16_t *distance_mm)
{
  uint8_t ready = 0U;
  HAL_StatusTypeDef st = tof_is_data_ready(&ready);
  if (st != HAL_OK)
  {
    return st;
  }
  if (ready == 0U)
  {
    return HAL_BUSY;
  }
  st = tof_read_u16(VL53L4CD_REG_RESULT_DISTANCE, distance_mm);
  if (st != HAL_OK)
  {
    return st;
  }
  return tof_write_u8(VL53L4CD_REG_SYSTEM_INTERRUPT_CLR, 0x01U);
}

static HAL_StatusTypeDef can_send_sensor_feedback(float body_angle,
                                                  float gyro_rate,
                                                  uint16_t tof_distance_mm)
{
  FDCAN_TxHeaderTypeDef tx_hdr = {0};
  uint8_t tx_data[8];
  uint16_t body_u16 = can_map_float_to_u16(body_angle, CAN_ANGLE_MIN, CAN_ANGLE_MAX);
  uint16_t gyro_u16 = can_map_float_to_u16(gyro_rate, CAN_RATE_MIN, CAN_RATE_MAX);

  tx_data[0] = (uint8_t)((SENSOR_STATE_RUN << 4) | (SENSOR_ID & 0x0FU));
  tx_data[1] = sensor_error_code;
  tx_data[2] = (uint8_t)((body_u16 >> 8) & 0xFFU);
  tx_data[3] = (uint8_t)(body_u16 & 0xFFU);
  tx_data[4] = (uint8_t)((gyro_u16 >> 8) & 0xFFU);
  tx_data[5] = (uint8_t)(gyro_u16 & 0xFFU);
  tx_data[6] = (uint8_t)((tof_distance_mm >> 8) & 0xFFU);
  tx_data[7] = (uint8_t)(tof_distance_mm & 0xFFU);

  tx_hdr.Identifier          = SENSOR_FEEDBACK_ID;
  tx_hdr.IdType              = FDCAN_STANDARD_ID;
  tx_hdr.TxFrameType         = FDCAN_DATA_FRAME;
  tx_hdr.DataLength          = FDCAN_DLC_BYTES_8;
  tx_hdr.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
  tx_hdr.BitRateSwitch       = FDCAN_BRS_OFF;
  tx_hdr.FDFormat            = FDCAN_CLASSIC_CAN;
  tx_hdr.TxEventFifoControl  = FDCAN_NO_TX_EVENTS;
  tx_hdr.MessageMarker       = 0;

  return HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan1, &tx_hdr, tx_data);
}

static HAL_StatusTypeDef can_send_sensor_aux_feedback(float body_angle_y,
                                                      float gyro_rate_y)
{
  FDCAN_TxHeaderTypeDef tx_hdr = {0};
  uint8_t tx_data[8];
  uint16_t body_y_u16 = can_map_float_to_u16(body_angle_y, CAN_ANGLE_MIN, CAN_ANGLE_MAX);
  uint16_t gyro_y_u16 = can_map_float_to_u16(gyro_rate_y, CAN_RATE_MIN, CAN_RATE_MAX);

  tx_data[0] = (uint8_t)((SENSOR_STATE_RUN << 4) | (SENSOR_ID & 0x0FU));
  tx_data[1] = sensor_error_code;
  tx_data[2] = (uint8_t)((body_y_u16 >> 8) & 0xFFU);
  tx_data[3] = (uint8_t)(body_y_u16 & 0xFFU);
  tx_data[4] = (uint8_t)((gyro_y_u16 >> 8) & 0xFFU);
  tx_data[5] = (uint8_t)(gyro_y_u16 & 0xFFU);
  tx_data[6] = 0U;
  tx_data[7] = 0U;

  tx_hdr.Identifier          = SENSOR_AUX_FEEDBACK_ID;
  tx_hdr.IdType              = FDCAN_STANDARD_ID;
  tx_hdr.TxFrameType         = FDCAN_DATA_FRAME;
  tx_hdr.DataLength          = FDCAN_DLC_BYTES_8;
  tx_hdr.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
  tx_hdr.BitRateSwitch       = FDCAN_BRS_OFF;
  tx_hdr.FDFormat            = FDCAN_CLASSIC_CAN;
  tx_hdr.TxEventFifoControl  = FDCAN_NO_TX_EVENTS;
  tx_hdr.MessageMarker       = 0;

  return HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan1, &tx_hdr, tx_data);
}

static HAL_StatusTypeDef can_send_sensor_accel_feedback(float accel_angle_x_value,
                                                        float accel_angle_y_value)
{
  FDCAN_TxHeaderTypeDef tx_hdr = {0};
  uint8_t tx_data[8];
  uint16_t accel_x_u16 = can_map_float_to_u16(accel_angle_x_value,
                                              CAN_ANGLE_MIN,
                                              CAN_ANGLE_MAX);
  uint16_t accel_y_u16 = can_map_float_to_u16(accel_angle_y_value,
                                              CAN_ANGLE_MIN,
                                              CAN_ANGLE_MAX);

  tx_data[0] = (uint8_t)((SENSOR_STATE_RUN << 4) | (SENSOR_ID & 0x0FU));
  tx_data[1] = (uint8_t)(accel_angles_valid |
                         ((gyro_reject_latched != 0U) ? 0x02U : 0x00U));
  tx_data[2] = (uint8_t)((accel_x_u16 >> 8) & 0xFFU);
  tx_data[3] = (uint8_t)(accel_x_u16 & 0xFFU);
  tx_data[4] = (uint8_t)((accel_y_u16 >> 8) & 0xFFU);
  tx_data[5] = (uint8_t)(accel_y_u16 & 0xFFU);
  tx_data[6] = 0U;
  tx_data[7] = 0U;

  tx_hdr.Identifier          = SENSOR_ACCEL_FEEDBACK_ID;
  tx_hdr.IdType              = FDCAN_STANDARD_ID;
  tx_hdr.TxFrameType         = FDCAN_DATA_FRAME;
  tx_hdr.DataLength          = FDCAN_DLC_BYTES_8;
  tx_hdr.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
  tx_hdr.BitRateSwitch       = FDCAN_BRS_OFF;
  tx_hdr.FDFormat            = FDCAN_CLASSIC_CAN;
  tx_hdr.TxEventFifoControl  = FDCAN_NO_TX_EVENTS;
  tx_hdr.MessageMarker       = 0;

  return HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan1, &tx_hdr, tx_data);
}

static HAL_StatusTypeDef can_send_sensor_yaw_feedback(float body_angle_z,
                                                       float gyro_rate_z)
{
  FDCAN_TxHeaderTypeDef tx_hdr = {0};
  uint8_t tx_data[8];
  uint16_t body_z_u16 = can_map_float_to_u16(body_angle_z, CAN_ANGLE_MIN, CAN_ANGLE_MAX);
  uint16_t gyro_z_u16 = can_map_float_to_u16(gyro_rate_z, CAN_RATE_MIN, CAN_RATE_MAX);

  tx_data[0] = (uint8_t)((SENSOR_STATE_RUN << 4) | (SENSOR_ID & 0x0FU));
  tx_data[1] = sensor_error_code;
  tx_data[2] = (uint8_t)((body_z_u16 >> 8) & 0xFFU);
  tx_data[3] = (uint8_t)(body_z_u16 & 0xFFU);
  tx_data[4] = (uint8_t)((gyro_z_u16 >> 8) & 0xFFU);
  tx_data[5] = (uint8_t)(gyro_z_u16 & 0xFFU);
  tx_data[6] = 0U;
  tx_data[7] = 0U;

  tx_hdr.Identifier          = SENSOR_YAW_FEEDBACK_ID;
  tx_hdr.IdType              = FDCAN_STANDARD_ID;
  tx_hdr.TxFrameType         = FDCAN_DATA_FRAME;
  tx_hdr.DataLength          = FDCAN_DLC_BYTES_8;
  tx_hdr.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
  tx_hdr.BitRateSwitch       = FDCAN_BRS_OFF;
  tx_hdr.FDFormat            = FDCAN_CLASSIC_CAN;
  tx_hdr.TxEventFifoControl  = FDCAN_NO_TX_EVENTS;
  tx_hdr.MessageMarker       = 0;

  return HAL_FDCAN_AddMessageToTxFifoQ(&hfdcan1, &tx_hdr, tx_data);
}

static void sensor_led_encode_byte(uint8_t value, uint32_t *index)
{
  uint8_t bit;
  for (bit = 0U; bit < 8U; bit++)
  {
    led_pwm_buf[*index] = ((value & 0x80U) != 0U) ? led_pwm_duty_1 : led_pwm_duty_0;
    (*index)++;
    value <<= 1U;
  }
}

static void sensor_led_set_all(uint8_t r, uint8_t g, uint8_t b)
{
  uint32_t led_i;
  uint32_t reset_i;
  uint32_t idx = 0U;
  uint32_t t0;

  for (led_i = 0U; led_i < SENSOR_LED_COUNT; led_i++)
  {
    /* WS2812 order is GRB. */
    sensor_led_encode_byte(g, &idx);
    sensor_led_encode_byte(r, &idx);
    sensor_led_encode_byte(b, &idx);
  }
  for (reset_i = 0U; reset_i < WS2812_RESET_SLOTS; reset_i++)
  {
    led_pwm_buf[idx++] = 0U;
  }

  led_dma_done = 0U;
  if (HAL_TIM_PWM_Start_DMA(&htim2, TIM_CHANNEL_2, (uint32_t *)led_pwm_buf, idx) != HAL_OK)
  {
    led_dma_done = 1U;
    return;
  }

  t0 = HAL_GetTick();
  while (led_dma_done == 0U)
  {
    if ((HAL_GetTick() - t0) > 10U)
    {
      HAL_TIM_PWM_Stop_DMA(&htim2, TIM_CHANNEL_2);
      led_dma_done = 1U;
      break;
    }
  }
}

static void sensor_led_init(void)
{
  uint32_t period_counts = (SystemCoreClock / WS2812_BITRATE_HZ);
  if (period_counts < 2U)
  {
    period_counts = 2U;
  }

  if (led_dma_initialized == 0U)
  {
    __HAL_RCC_DMAMUX1_CLK_ENABLE();
    __HAL_RCC_DMA1_CLK_ENABLE();

    hdma_tim2_ch2.Instance = DMA1_Channel1;
    hdma_tim2_ch2.Init.Request = DMA_REQUEST_TIM2_CH2;
    hdma_tim2_ch2.Init.Direction = DMA_MEMORY_TO_PERIPH;
    hdma_tim2_ch2.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_tim2_ch2.Init.MemInc = DMA_MINC_ENABLE;
    hdma_tim2_ch2.Init.PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD;
    hdma_tim2_ch2.Init.MemDataAlignment = DMA_MDATAALIGN_HALFWORD;
    hdma_tim2_ch2.Init.Mode = DMA_NORMAL;
    hdma_tim2_ch2.Init.Priority = DMA_PRIORITY_HIGH;
    if (HAL_DMA_Init(&hdma_tim2_ch2) != HAL_OK)
    {
      Error_Handler();
    }
    __HAL_LINKDMA(&htim2, hdma[TIM_DMA_ID_CC2], hdma_tim2_ch2);

    HAL_NVIC_SetPriority(DMA1_Channel1_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(DMA1_Channel1_IRQn);
    led_dma_initialized = 1U;
  }

  __HAL_TIM_SET_PRESCALER(&htim2, 0U);
  __HAL_TIM_SET_AUTORELOAD(&htim2, period_counts - 1U);
  __HAL_TIM_SET_COMPARE(&htim2, TIM_CHANNEL_2, 0U);
  HAL_TIM_GenerateEvent(&htim2, TIM_EVENTSOURCE_UPDATE);

  led_pwm_duty_0 = (uint16_t)((period_counts * 35U) / 100U);
  led_pwm_duty_1 = (uint16_t)((period_counts * 70U) / 100U);
  if (led_pwm_duty_0 == 0U)
  {
    led_pwm_duty_0 = 1U;
  }
  if (led_pwm_duty_1 <= led_pwm_duty_0)
  {
    led_pwm_duty_1 = led_pwm_duty_0 + 1U;
  }
}

void HAL_TIM_PWM_PulseFinishedCallback(TIM_HandleTypeDef *htim)
{
  if (htim->Instance == TIM2)
  {
    HAL_TIM_PWM_Stop_DMA(&htim2, TIM_CHANNEL_2);
    led_dma_done = 1U;
  }
}

void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t RxFifo0ITs)
{
  FDCAN_RxHeaderTypeDef rx_hdr;
  uint8_t rx_data[8];
  uint32_t now_ms;

  if ((RxFifo0ITs & FDCAN_IT_RX_FIFO0_NEW_MESSAGE) == 0U)
  {
    return;
  }

  if (HAL_FDCAN_GetRxMessage(hfdcan, FDCAN_RX_FIFO0, &rx_hdr, rx_data) != HAL_OK)
  {
    return;
  }

  can_rx_count++;
  now_ms = HAL_GetTick();

  if (rx_hdr.RxFrameType != FDCAN_DATA_FRAME)
  {
    return;
  }

  if (rx_hdr.Identifier == FLYWHEEL_X_CTRL_ID)
  {
    uint32_t override_ms = (rx_data[0] == CMD_MOTOR_ALIGN) ? MOTOR_ALIGN_OVERRIDE_MS : MOTOR_CMD_OVERRIDE_MS;
    flywheel_x_override_until_ms = now_ms + override_ms;
    if (rx_data[0] == CMD_MOTOR_START)
    {
      flywheel_x_enabled = 1U;
    }
    else if (rx_data[0] == CMD_MOTOR_STOP)
    {
      flywheel_x_enabled = 0U;
    }
  }
  else if (rx_hdr.Identifier == FLYWHEEL_Y_CTRL_ID)
  {
    uint32_t override_ms = (rx_data[0] == CMD_MOTOR_ALIGN) ? MOTOR_ALIGN_OVERRIDE_MS : MOTOR_CMD_OVERRIDE_MS;
    flywheel_y_override_until_ms = now_ms + override_ms;
    if (rx_data[0] == CMD_MOTOR_START)
    {
      flywheel_y_enabled = 1U;
    }
    else if (rx_data[0] == CMD_MOTOR_STOP)
    {
      flywheel_y_enabled = 0U;
    }
  }

  if (rx_hdr.DataLength != FDCAN_DLC_BYTES_8)
  {
    return;
  }

  if (rx_hdr.Identifier == FLYWHEEL_X_FEEDBACK_ID)
  {
    can_unpack_motor_feedback(rx_data, (CAN_MotorFeedback_t *)&flywheel_x_feedback);
  }
  else if (rx_hdr.Identifier == FLYWHEEL_Y_FEEDBACK_ID)
  {
    can_unpack_motor_feedback(rx_data, (CAN_MotorFeedback_t *)&flywheel_y_feedback);
  }
  else if ((rx_hdr.Identifier == SENSOR_ID || rx_hdr.Identifier == SENSOR_CTRL_ID) &&
           rx_data[0] == SENSOR_LED_CMD_MAGIC)
  {
#if (SENSOR_LED_TEST_CYCLE_ENABLE == 0U)
    led_r_cmd = rx_data[1];
    led_g_cmd = rx_data[2];
    led_b_cmd = rx_data[3];
    led_update_pending = 1U;
#endif
  }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  /* Force the vector table to the application's table in flash. Some boot
   * configurations leave SCB->VTOR at its reset value (0x00000000), which on
   * this board is NOT aliased to app flash, so every interrupt (SysTick,
   * FDCAN, ...) vectors into the wrong table and never runs. Setting VTOR
   * explicitly guarantees our handlers execute regardless of boot mapping. */
  SCB->VTOR = FLASH_BASE;
  __DSB();
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_I2C1_Init();
  MX_LPUART1_UART_Init();
  MX_USART1_UART_Init();
  MX_SPI1_Init();
  MX_TIM2_Init();
  MX_FDCAN1_Init();
  /* USER CODE BEGIN 2 */
  sensor_led_init();
  sensor_led_set_all(0U, 0U, 0U);

  /* ── IMU init ─────────────────────────────────────────────────────────── */
  CS_IMU_DEASSERT();
  //HAL_Delay(10);

  imu_ctx.write_reg = imu_spi_write;
  imu_ctx.read_reg  = imu_spi_read;
  imu_ctx.mdelay    = HAL_Delay;
  imu_ctx.handle    = &hspi1;

  uint8_t who_am_i = 0;
  ism330dhcx_device_id_get(&imu_ctx, &who_am_i);
  if (who_am_i == ISM330DHCX_ID)
  {
    sensor_error_code = 0U;
    uart_print("ISM330DHCX OK (0x6B)\r\n");
  }
  else
  {
    sensor_error_code = SENSOR_ERROR_IMU_ID;
    snprintf(uart_buf, sizeof(uart_buf), "ISM330DHCX FAIL: got 0x%02X\r\n", who_am_i);
    uart_print(uart_buf);
  }

  {
    int32_t imu_config_status = 0;
    imu_config_status |= ism330dhcx_xl_data_rate_set(&imu_ctx, ISM330DHCX_XL_ODR_1666Hz);
    imu_config_status |= ism330dhcx_xl_full_scale_set(&imu_ctx, ISM330DHCX_4g);
    imu_config_status |= ism330dhcx_gy_data_rate_set(&imu_ctx, ISM330DHCX_GY_ODR_1666Hz);
    imu_config_status |= ism330dhcx_gy_full_scale_set(&imu_ctx, ISM330DHCX_2000dps);
    imu_config_status |= ism330dhcx_block_data_update_set(&imu_ctx, PROPERTY_ENABLE);
    imu_config_status |= ism330dhcx_gy_filter_lp1_set(&imu_ctx, PROPERTY_ENABLE);
    imu_config_status |= ism330dhcx_gy_lp1_bandwidth_set(&imu_ctx, ISM330DHCX_MEDIUM);
    if (imu_config_status != 0)
    {
      sensor_error_code |= SENSOR_ERROR_IMU_DATA;
    }
    else
    {
      imu_config_valid = 1U;
    }
  }

  /* ── FDCAN init ───────────────────────────────────────────────────────── */
  /* Match the flywheel/spring_actuator boards' proven filter setup:
   * a range filter that accepts every standard ID into RX FIFO 0, and let
   * non-matching frames also fall through to FIFO 0 (HW default). Routing
   * non-matching frames to FIFO 1 here previously hid all RX traffic. */
  FDCAN_FilterTypeDef can_filter = {0};
  can_filter.IdType       = FDCAN_STANDARD_ID;
  can_filter.FilterIndex  = 0;
  can_filter.FilterType   = FDCAN_FILTER_RANGE;
  can_filter.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
  can_filter.FilterID1    = 0x000;   /* accept all IDs */
  can_filter.FilterID2    = 0x7FF;

  if (HAL_FDCAN_ConfigFilter(&hfdcan1, &can_filter) != HAL_OK) Error_Handler();
  if (HAL_FDCAN_ConfigGlobalFilter(&hfdcan1,
        FDCAN_ACCEPT_IN_RX_FIFO0, FDCAN_ACCEPT_IN_RX_FIFO0,
        FDCAN_FILTER_REMOTE, FDCAN_FILTER_REMOTE) != HAL_OK) Error_Handler();
  if (HAL_FDCAN_Start(&hfdcan1) != HAL_OK) Error_Handler();
  if (HAL_FDCAN_ActivateNotification(&hfdcan1, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0) != HAL_OK) Error_Handler();

  uart_print("FDCAN started.\r\n");

  if (tof_init() == HAL_OK)
  {
    sensor_error_code &= (uint8_t)~(SENSOR_ERROR_TOF_INIT | SENSOR_ERROR_TOF_DATA);
    uart_print("VL53L4CD OK.\r\n");
  }
  else
  {
    sensor_error_code |= (SENSOR_ERROR_TOF_INIT | SENSOR_ERROR_TOF_DATA);
    uart_print("VL53L4CD init failed.\r\n");
  }

  uart_print("Flywheel control idle until external ON command.\r\n");

  /* Gravity points along -Z when upright, while the accelerometer reports
   * opposite-gravity specific force along +Z. */
  {
    int16_t raw_a[3] = {0};
    if (ism330dhcx_acceleration_raw_get(&imu_ctx, raw_a) == 0)
    {
      float ax0_g = ism330dhcx_from_fs4g_to_mg(raw_a[0]) * 0.001f;
      float ay0_g = ism330dhcx_from_fs4g_to_mg(raw_a[1]) * 0.001f;
      float az0_g = ism330dhcx_from_fs4g_to_mg(raw_a[2]) * 0.001f;
      accel_lpf_x_g = ax0_g;
      accel_lpf_y_g = ay0_g;
      accel_lpf_z_g = az0_g;
      accel_angle_x = atan2f(-ay0_g, az0_g);
      accel_angle_y = atan2f(ax0_g, sqrtf(ay0_g * ay0_g + az0_g * az0_g));
      theta_x = accel_angle_x;
      theta_y = accel_angle_y;
    }
    else
    {
      sensor_error_code |= SENSOR_ERROR_IMU_DATA;
    }
    madgwick_set_tilt(theta_x, theta_y);
  }
  uart_print("Init done.\r\n\r\n");

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    /* ── Read IMU ─────────────────────────────────────────────────────── */
    int16_t raw_accel[3] = {0};
    int16_t raw_gyro[3] = {0};
    int32_t accel_read_status = ism330dhcx_acceleration_raw_get(&imu_ctx, raw_accel);
    int32_t gyro_read_status = ism330dhcx_angular_rate_raw_get(&imu_ctx, raw_gyro);
    uint8_t imu_sample_valid =
        (accel_read_status == 0 && gyro_read_status == 0) ? 1U : 0U;

    float ax = ism330dhcx_from_fs4g_to_mg(raw_accel[0]);
    float ay = ism330dhcx_from_fs4g_to_mg(raw_accel[1]);
    float az = ism330dhcx_from_fs4g_to_mg(raw_accel[2]);
    float gx = ism330dhcx_from_fs2000dps_to_mdps(raw_gyro[0]);
    float gy = ism330dhcx_from_fs2000dps_to_mdps(raw_gyro[1]);
    float gz = ism330dhcx_from_fs2000dps_to_mdps(raw_gyro[2]);

    /* ── Dual-axis state update + LQR control ─────────────────────────── */
    uint32_t now = HAL_GetTick();
    float dt = (ctrl_t_prev == 0U) ? 0.001f : (float)(now - ctrl_t_prev) * 0.001f;
    if (dt > 0.1f) dt = 0.001f;
    ctrl_t_prev = now;

    if ((now - can_health_prev) >= CAN_HEALTH_PERIOD_MS)
    {
      FDCAN_ProtocolStatusTypeDef protocol_status = {0};
      can_health_prev = now;
      if (HAL_FDCAN_GetProtocolStatus(&hfdcan1, &protocol_status) == HAL_OK &&
          protocol_status.BusOff != 0U)
      {
        sensor_error_code |= SENSOR_ERROR_CAN_TX;
        if (HAL_FDCAN_Stop(&hfdcan1) == HAL_OK &&
            HAL_FDCAN_Start(&hfdcan1) == HAL_OK &&
            HAL_FDCAN_ActivateNotification(&hfdcan1,
                                            FDCAN_IT_RX_FIFO0_NEW_MESSAGE,
                                            0U) == HAL_OK)
        {
          can_tx_prev = now;
          can_ctrl_prev = now;
        }
      }
    }

    /* Rotate the physical IMU frame 180 degrees about Z into the controller
     * frame used by the existing X/Y angle and motor sign conventions. */
    float gx_raw_rads = -gx * (1.0f / 1000.0f) * (3.14159265f / 180.0f);
    float gy_raw_rads = -gy * (1.0f / 1000.0f) * (3.14159265f / 180.0f);
    float gz_raw_rads =  gz * (1.0f / 1000.0f) * (3.14159265f / 180.0f);
    float gx_rads;
    float gy_rads;
    float gz_rads;
    uint8_t gyro_sample_valid = gyro_filter_update(gx_raw_rads,
                                                    gy_raw_rads,
                                                    gz_raw_rads,
                                                    dt,
                                                    imu_sample_valid,
                                                    &gx_rads,
                                                    &gy_rads,
                                                    &gz_rads);
    if (gyro_sample_valid != 0U)
    {
      if (gyro_recovery_count < GYRO_RECOVERY_SAMPLES)
      {
        gyro_recovery_count++;
      }
      if (gyro_recovery_count >= GYRO_RECOVERY_SAMPLES &&
          imu_config_valid != 0U)
      {
        sensor_error_code &= (uint8_t)~SENSOR_ERROR_IMU_DATA;
      }
    }
    else
    {
      gyro_recovery_count = 0U;
      sensor_error_code |= SENSOR_ERROR_IMU_DATA;
    }

    {
      float ax_g = ax * 0.001f;
      float ay_g = ay * 0.001f;
      float az_g = az * 0.001f;
      float raw_mag = sqrtf(ax_g * ax_g + ay_g * ay_g + az_g * az_g);
      float lpf_alpha = (dt > 0.0f) ? (dt / (ACCEL_LPF_TAU_S + dt)) : 0.0f;
      accel_lpf_x_g += lpf_alpha * (ax_g - accel_lpf_x_g);
      accel_lpf_y_g += lpf_alpha * (ay_g - accel_lpf_y_g);
      accel_lpf_z_g += lpf_alpha * (az_g - accel_lpf_z_g);

      float filtered_mag = sqrtf(accel_lpf_x_g * accel_lpf_x_g +
                                 accel_lpf_y_g * accel_lpf_y_g +
                                 accel_lpf_z_g * accel_lpf_z_g);
      if (imu_sample_valid != 0U &&
          raw_mag >= ACCEL_RAW_MAG_LO && raw_mag <= ACCEL_RAW_MAG_HI &&
          filtered_mag >= ACCEL_FILTERED_MAG_LO &&
          filtered_mag <= ACCEL_FILTERED_MAG_HI)
      {
        if (accel_valid_count < ACCEL_VALID_SAMPLES)
        {
          accel_valid_count++;
        }
      }
      else
      {
        accel_valid_count = 0U;
      }

      accel_angle_x = atan2f(-accel_lpf_y_g, accel_lpf_z_g);
      accel_angle_y = atan2f(accel_lpf_x_g,
                             sqrtf(accel_lpf_y_g * accel_lpf_y_g +
                                   accel_lpf_z_g * accel_lpf_z_g));
    }

    accel_angles_valid =
        (accel_valid_count >= ACCEL_VALID_SAMPLES &&
         fabsf(gx_rads) <= ACCEL_RATE_LIMIT_RAD_S &&
         fabsf(gy_rads) <= ACCEL_RATE_LIMIT_RAD_S) ? 1U : 0U;
    madgwick_update_imu(gx_rads,
                        gy_rads,
                        gz_rads,
                        -accel_lpf_x_g,
                        -accel_lpf_y_g,
                        accel_lpf_z_g,
                        dt,
                        accel_angles_valid);
    madgwick_get_attitude(&theta_x, &theta_y, &theta_z);

    theta_x = wrap_to_pi(theta_x);
    theta_y = wrap_to_pi(theta_y);
    theta_z = wrap_to_pi(theta_z);
    float angle_err_x = wrap_to_pi(theta_x - LQR_THETA_SETPOINT_X);
    float angle_err_y = wrap_to_pi(theta_y - LQR_THETA_SETPOINT_Y);
    float flywheel_x_vel;
    float flywheel_y_vel;
    __disable_irq();
    flywheel_x_vel = flywheel_x_feedback.velocity_actual;
    flywheel_y_vel = flywheel_y_feedback.velocity_actual;
    __enable_irq();

    float torque_cmd_x = LQR_X_K1 * angle_err_x + LQR_X_K2 * gx_rads + LQR_X_K3 * flywheel_x_vel;
    float torque_cmd_y = LQR_Y_K1 * angle_err_y + LQR_Y_K2 * gy_rads + LQR_Y_K3 * flywheel_y_vel;
    if (torque_cmd_x > LQR_TORQUE_LIMIT) torque_cmd_x = LQR_TORQUE_LIMIT;
    if (torque_cmd_x < -LQR_TORQUE_LIMIT) torque_cmd_x = -LQR_TORQUE_LIMIT;
    if (torque_cmd_y > LQR_TORQUE_LIMIT) torque_cmd_y = LQR_TORQUE_LIMIT;
    if (torque_cmd_y < -LQR_TORQUE_LIMIT) torque_cmd_y = -LQR_TORQUE_LIMIT;

    {
      uint16_t distance_mm = 0U;
      HAL_StatusTypeDef tof_read_status = tof_try_read_distance(&distance_mm);
      if (tof_read_status == HAL_OK)
      {
        tof_distance_mm = distance_mm;
        sensor_error_code &= (uint8_t)~SENSOR_ERROR_TOF_DATA;
      }
      else if (tof_read_status != HAL_BUSY)
      {
        sensor_error_code |= SENSOR_ERROR_TOF_DATA;
      }
    }

    if ((now - can_tx_prev) >= CAN_TX_PERIOD_MS)
    {
      HAL_StatusTypeDef aux_tx_status;
      HAL_StatusTypeDef accel_tx_status;
      HAL_StatusTypeDef yaw_tx_status;
      uint32_t tx_free;
      can_tx_prev = now;
      tx_free = HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan1);
      if (tx_free > 0U)
      {
        can_last_tx_status = can_send_sensor_feedback(theta_x, gx_rads, tof_distance_mm);
      }
      else
      {
        can_last_tx_status = HAL_BUSY;
      }

      tx_free = HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan1);
      if (tx_free > 0U)
      {
        aux_tx_status = can_send_sensor_aux_feedback(theta_y, gy_rads);
      }
      else
      {
        aux_tx_status = HAL_BUSY;
      }

      tx_free = HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan1);
      if (tx_free > 0U)
      {
        accel_tx_status = can_send_sensor_accel_feedback(accel_angle_x,
                                                         accel_angle_y);
        if (accel_tx_status == HAL_OK)
        {
          gyro_reject_latched = 0U;
        }
      }
      else
      {
        accel_tx_status = HAL_BUSY;
      }
      tx_free = HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan1);
      if (tx_free > 0U)
      {
        yaw_tx_status = can_send_sensor_yaw_feedback(theta_z, gz_rads);
      }
      else
      {
        yaw_tx_status = HAL_BUSY;
      }
      if (can_last_tx_status == HAL_OK && aux_tx_status != HAL_OK && aux_tx_status != HAL_BUSY)
      {
        can_last_tx_status = aux_tx_status;
      }
      if (can_last_tx_status == HAL_OK &&
          accel_tx_status != HAL_OK &&
          accel_tx_status != HAL_BUSY)
      {
        can_last_tx_status = accel_tx_status;
      }
      if (can_last_tx_status == HAL_OK &&
          yaw_tx_status != HAL_OK &&
          yaw_tx_status != HAL_BUSY)
      {
        can_last_tx_status = yaw_tx_status;
      }
      if (can_last_tx_status != HAL_OK && can_last_tx_status != HAL_BUSY)
      {
        sensor_error_code |= SENSOR_ERROR_CAN_TX;
      }
      else if (can_last_ctrl_status == HAL_OK || can_last_ctrl_status == HAL_BUSY)
      {
        sensor_error_code &= (uint8_t)~SENSOR_ERROR_CAN_TX;
      }
    }

    if ((now - can_ctrl_prev) >= CAN_CTRL_PERIOD_MS)
    {
      CAN_Command_t cmd_x = {0};
      CAN_Command_t cmd_y = {0};
      HAL_StatusTypeDef tx_x = HAL_BUSY;
      HAL_StatusTypeDef tx_y = HAL_BUSY;
      uint32_t x_override_until;
      uint32_t y_override_until;
      uint8_t x_enabled;
      uint8_t y_enabled;
      can_ctrl_prev = now;
      __disable_irq();
      x_override_until = flywheel_x_override_until_ms;
      y_override_until = flywheel_y_override_until_ms;
      x_enabled = flywheel_x_enabled;
      y_enabled = flywheel_y_enabled;
      __enable_irq();
      if (x_enabled != 0U &&
          (now >= x_override_until) &&
          (HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan1) > 0U))
      {
        cmd_x.torque_ff = torque_cmd_x;
        tx_x = can_send_motor_command(FLYWHEEL_X_ID, &cmd_x);
      }
      if (y_enabled != 0U &&
          (now >= y_override_until) &&
          (HAL_FDCAN_GetTxFifoFreeLevel(&hfdcan1) > 0U))
      {
        cmd_y.torque_ff = torque_cmd_y;
        tx_y = can_send_motor_command(FLYWHEEL_Y_ID, &cmd_y);
      }
      can_last_ctrl_status = ((tx_x == HAL_OK || x_enabled == 0U || now < x_override_until) &&
                              (tx_y == HAL_OK || y_enabled == 0U || now < y_override_until)) ? HAL_OK : HAL_BUSY;
      if ((tx_x != HAL_OK && tx_x != HAL_BUSY) || (tx_y != HAL_OK && tx_y != HAL_BUSY))
      {
        sensor_error_code |= SENSOR_ERROR_CAN_TX;
        can_last_ctrl_status = HAL_ERROR;
      }
    }

    if (led_update_pending != 0U)
    {
      uint8_t r, g, b;
      __disable_irq();
      r = led_r_cmd;
      g = led_g_cmd;
      b = led_b_cmd;
      led_update_pending = 0U;
      __enable_irq();
      sensor_led_set_all(r, g, b);
    }

#if (SENSOR_LED_TEST_CYCLE_ENABLE == 1U)
    if ((now - led_test_prev_ms) >= SENSOR_LED_TEST_PERIOD_MS)
    {
      static const uint8_t test_colors[][3] = {
        {255U,   0U,   0U},
        {  0U, 255U,   0U},
        {  0U,   0U, 255U},
        {255U, 128U,   0U},
        {255U,   0U, 255U},
        {  0U, 255U, 255U},
        {255U, 255U, 255U},
        {  0U,   0U,   0U},
      };
      led_test_prev_ms = now;
      sensor_led_set_all(test_colors[led_test_idx][0],
                         test_colors[led_test_idx][1],
                         test_colors[led_test_idx][2]);
      led_test_idx++;
      if (led_test_idx >= (uint8_t)(sizeof(test_colors) / sizeof(test_colors[0])))
      {
        led_test_idx = 0U;
      }
    }
#endif

#if (UART_PERIODIC_DEBUG_ENABLE == 1U)
    float accel_mag = sqrtf((ax*ax + ay*ay + az*az) * 1e-6f);
    /* ── Loop rate tracking ───────────────────────────────────────────── */
    loop_count++;
    uint32_t loop_now = HAL_GetTick();

    /* ── Print @ ~10 Hz ───────────────────────────────────────────────── */
    print_counter++;
    if (print_counter >= 100U)
    {
      print_counter = 0;
      uint32_t elapsed = loop_now - print_t_prev;
      float loop_hz = (elapsed > 0U) ? ((float)loop_count * 1000.0f / (float)elapsed) : 0.0f;
      loop_count   = 0;
      print_t_prev = loop_now;

      snprintf(uart_buf, sizeof(uart_buf),
               "%.0fHz  thx=%.3f thy=%.3f gx=%.2f gy=%.2f fwx=%.1f fwy=%.1f tx=%.3f ty=%.3f tof=%umm |a|=%.3f\r\n",
               (double)loop_hz,
               (double)theta_x,
               (double)theta_y,
               (double)gx_rads,
               (double)gy_rads,
               (double)flywheel_x_vel,
               (double)flywheel_y_vel,
               (double)torque_cmd_x,
               (double)torque_cmd_y,
               (unsigned int)tof_distance_mm,
               (double)accel_mag);
      uart_print(uart_buf);

      /* ── CAN health readout ─────────────────────────────────────────────
       * TXok = last queue status (0 = HAL_OK); TEC/REC = error counters;
       * RX = frames received from the flywheel feedback ID. */
      FDCAN_ProtocolStatusTypeDef ps = {0};
      FDCAN_ErrorCountersTypeDef  ec = {0};
      HAL_FDCAN_GetProtocolStatus(&hfdcan1, &ps);
      HAL_FDCAN_GetErrorCounters(&hfdcan1, &ec);
      snprintf(uart_buf, sizeof(uart_buf),
               "  CAN sensorTX=%d ctrlTX=%d BusOff=%lu TEC=%lu REC=%lu RX=%lu\r\n",
               (int)can_last_tx_status,
               (int)can_last_ctrl_status,
               (unsigned long)ps.BusOff,
               (unsigned long)ec.TxErrorCnt,
               (unsigned long)ec.RxErrorCnt,
               (unsigned long)can_rx_count);
      uart_print(uart_buf);
    }
#endif
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief FDCAN1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_FDCAN1_Init(void)
{

  /* USER CODE BEGIN FDCAN1_Init 0 */

  /* USER CODE END FDCAN1_Init 0 */

  /* USER CODE BEGIN FDCAN1_Init 1 */

  /* USER CODE END FDCAN1_Init 1 */
  hfdcan1.Instance = FDCAN1;
  hfdcan1.Init.ClockDivider = FDCAN_CLOCK_DIV1;
  hfdcan1.Init.FrameFormat = FDCAN_FRAME_CLASSIC;
  hfdcan1.Init.Mode = FDCAN_MODE_NORMAL;
  hfdcan1.Init.AutoRetransmission = DISABLE;
  hfdcan1.Init.TransmitPause = DISABLE;
  hfdcan1.Init.ProtocolException = DISABLE;
  hfdcan1.Init.NominalPrescaler = 1;
  hfdcan1.Init.NominalSyncJumpWidth = 1;
  hfdcan1.Init.NominalTimeSeg1 = 12;
  hfdcan1.Init.NominalTimeSeg2 = 3;
  hfdcan1.Init.DataPrescaler = 1;
  hfdcan1.Init.DataSyncJumpWidth = 1;
  hfdcan1.Init.DataTimeSeg1 = 1;
  hfdcan1.Init.DataTimeSeg2 = 1;
  hfdcan1.Init.StdFiltersNbr = 0;
  hfdcan1.Init.ExtFiltersNbr = 0;
  hfdcan1.Init.TxFifoQueueMode = FDCAN_TX_FIFO_OPERATION;
  if (HAL_FDCAN_Init(&hfdcan1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN FDCAN1_Init 2 */

  /* USER CODE END FDCAN1_Init 2 */

}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.Timing = 0x00503D58;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief LPUART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_LPUART1_UART_Init(void)
{

  /* USER CODE BEGIN LPUART1_Init 0 *x/

  /* USER CODE END LPUART1_Init 0 */

  /* USER CODE BEGIN LPUART1_Init 1 */

  /* USER CODE END LPUART1_Init 1 */
  hlpuart1.Instance = LPUART1;
  hlpuart1.Init.BaudRate = 115200;
  hlpuart1.Init.WordLength = UART_WORDLENGTH_8B;
  hlpuart1.Init.StopBits = UART_STOPBITS_1;
  hlpuart1.Init.Parity = UART_PARITY_NONE;
  hlpuart1.Init.Mode = UART_MODE_TX_RX;
  hlpuart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  hlpuart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  hlpuart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  hlpuart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&hlpuart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&hlpuart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&hlpuart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&hlpuart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN LPUART1_Init 2 */

  /* USER CODE END LPUART1_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 7;
  hspi1.Init.CRCLength = SPI_CRC_LENGTH_DATASIZE;
  hspi1.Init.NSSPMode = SPI_NSS_PULSE_DISABLE;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 4294967295;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, CS_ACCEL_Pin|CS_MAG_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(INT1_GPIO_Port, INT1_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pins : CS_ACCEL_Pin CS_MAG_Pin */
  GPIO_InitStruct.Pin = CS_ACCEL_Pin|CS_MAG_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : INT1_Pin */
  GPIO_InitStruct.Pin = INT1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(INT1_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  __disable_irq();
  while (1) {}
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
