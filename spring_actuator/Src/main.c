/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <stdint.h>
#include "mc_api.h"
#include "mc_config.h"
#include "mc_type.h"
#include "mc_config_common.h"
#include "encoder_speed_pos_fdbk.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct {
    float position_target; /* rad */
    float velocity_target; /* rad/s */
    float kp;              /* Nm/rad */
    float kd;              /* Nm/(rad/s) */
    float torque_ff;       /* Nm */
} CAN_Command_t;

typedef struct {
    float position_actual;
    float velocity_actual;
    float torque_actual;
    uint8_t motor_id;
    uint8_t error_code;
} CAN_Feedback_t;

typedef struct {
    float position_target;
    float velocity_target;
    float torque_target;
    float kp;
    float kd;

    float position_curr;
    float velocity_curr;
    float torque_curr;

    float position_step;
    float velocity_step;
    float torque_step;

    uint16_t steps_per_can_frame;
    uint16_t interpolation_counter;
} ImpedanceController_t;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define MA732_SPI_TIMEOUT_MS  10U
#define MA732_COUNTS_PER_TURN 16384U
#define MA732_COUNTS_MASK     (MA732_COUNTS_PER_TURN - 1U)
#define MA732_DPP_PER_COUNT   (65536U / MA732_COUNTS_PER_TURN)
#define MA732_DIR_INVERT      0U
#define MA732_ELEC_OFFSET_COUNTS 0

/* ---- CAN node identity ------------------------------------------------- */
#define MY_MOTOR_ID   0x03              /* spring_actuator node; flywheel_x=0x01, flywheel_y=0x02 */
#define CTRL_ID       (MY_MOTOR_ID + 0x200)  /* state-machine commands from host */
#define CMD_MOTOR_START  0x01
#define CMD_MOTOR_STOP   0x02
#define CMD_FEEDBACK_REQUEST 0x04
#define CMD_FORCE_ALIGN  0x03   /* stop, clear encoder alignment, re-run */

/* ---- Motor physical parameters ----------------------------------------- */
/* From motor datasheet */
#define MOTOR_KT          0.039f    /* Nm/A */
/* Hardware maximum readable current = 3.3 / (2 × RSHUNT × AMP_GAIN)
 *   = 3.3 / (2 × 0.01 × 7.33) = 22.5 A peak.
 * IQMAX_A = 20 A (drive_parameters.h) keeps int16 safe:
 *   20 × CURRENT_CONV_FACTOR(1456) = 29120 < 32767.
 * IQ_MAX_A = 0.76 / 0.039 = 19.5 A < 20 A — within both limits. */
#define TORQUE_MAX_NM     0.76f     /* Nm, from datasheet */
#define IQ_MAX_A          (TORQUE_MAX_NM / MOTOR_KT)  /* ~19.5 A */
#define MOTOR_POLE_PAIRS  POLE_PAIR_NUM

/* ---- Velocity sign convention ------------------------------------------ */
/* Set to -1.0f if reported velocity is backwards relative to positive torque.
 * Positive direction: turning the shaft by hand in the commanded direction
 * should show a positive velocity_actual in the CAN feedback. */
#define ENCODER_VEL_SIGN  1.0f

/* ---- Unit conversion --------------------------------------------------- */
#define S16_TO_RADIANS  (3.14159265f / 32768.0f)
#define RPM_TO_RADS     0.104719755f
#ifndef M_PI
#define M_PI            3.14159265f
#endif

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
ADC_HandleTypeDef hadc2;

CORDIC_HandleTypeDef hcordic;

FDCAN_HandleTypeDef hfdcan1;

I2C_HandleTypeDef hi2c3;

OPAMP_HandleTypeDef hopamp1;
OPAMP_HandleTypeDef hopamp2;

SPI_HandleTypeDef hspi1;

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim6;   /* MA732 background refresh timer (20 kHz) */

UART_HandleTypeDef huart1;
DMA_HandleTypeDef hdma_usart1_rx;
DMA_HandleTypeDef hdma_usart1_tx;

/* USER CODE BEGIN PV */
static volatile uint16_t ma732_raw14_cache = 0U;
static volatile uint8_t ma732_cache_valid = 0U;
static volatile int32_t ma732_elec_accum_counts = 0;
static volatile uint16_t ma732_last_raw14_for_pos = 0U;
static volatile uint8_t ma732_pos_initialized = 0U;
static volatile int32_t ma732_elec_offset_counts = MA732_ELEC_OFFSET_COUNTS;

/* Impedance / CAN state
 * NOTE: hfdcan1 and MX_FDCAN1_Init() are generated by CubeMX once you enable
 * FDCAN1 in the .ioc.  Remove the hfdcan1 line below after CubeMX regenerates. */
FDCAN_HandleTypeDef   hfdcan1;           /* TEMP: remove after CubeMX generates */
CAN_Command_t         IncomingCmd   = {0};
CAN_Feedback_t        CurrentState  = {0};
ImpedanceController_t ImpedanceCtrl = {0};
volatile uint32_t     can_rx_count  = 0;
volatile uint8_t      can_active    = 0; /* set on first valid CAN command */
volatile uint8_t      force_align   = 0; /* set by CAN CMD_FORCE_ALIGN; handled in main loop */
volatile uint16_t     dbg_ma732_raw14 = 0U;
volatile uint8_t      dbg_ma732_valid = 0U;
volatile int16_t      dbg_mec_angle = 0;
volatile int16_t      dbg_el_angle = 0;
volatile int16_t      dbg_mc_state = 0;
volatile uint32_t     dbg_ma732_irq_count = 0U;
volatile uint32_t     dbg_ma732_read_ok_count = 0U;
volatile uint32_t     dbg_ma732_read_fail_count = 0U;
volatile uint8_t      dbg_ma732_last_read_ok = 0U;
volatile uint16_t     dbg_ma732_last_read_raw14 = 0U;
volatile uint16_t     dbg_ma732_last_raw_word = 0U;
volatile uint16_t     dbg_ma732_last_raw14_msb = 0U;
volatile uint16_t     dbg_ma732_last_raw14_lsb = 0U;
volatile uint8_t      dbg_ma732_last_spi_status = 0U;
volatile uint32_t     dbg_ma732_last_spi_error = 0U;
volatile uint32_t     dbg_ma732_last_spi_sr = 0U;
volatile uint8_t      dbg_ma732_last_rx0 = 0U;
volatile uint8_t      dbg_ma732_last_rx1 = 0U;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_ADC1_Init(void);
static void MX_ADC2_Init(void);
static void MX_CORDIC_Init(void);
static void MX_I2C3_Init(void);
static void MX_OPAMP1_Init(void);
static void MX_OPAMP2_Init(void);
static void MX_TIM1_Init(void);
static void MX_TIM4_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_SPI1_Init(void);
static void MX_FDCAN1_Init(void);
static void MX_NVIC_Init(void);
/* USER CODE BEGIN PFP */
/* NOTE: MX_FDCAN1_Init() prototype will be added by CubeMX — do not add here */
void CAN_Interrupt_Init(void);
void MA732_TimerInit(void);
float Math_Map_Int_To_Float(int x_int, float x_min, float x_max, int bits);
int   Math_Map_Float_To_Int(float x, float x_min, float x_max, int bits);
void  CAN_Unpack_Command(const uint8_t *data, CAN_Command_t *cmd);
void  CAN_Pack_Feedback(uint8_t *data, const CAN_Feedback_t *fb);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static bool MA732_ReadRaw14(uint16_t *raw14)
{
  uint8_t tx_data[2] = {0x00U, 0x00U};
  uint8_t rx_data[2] = {0x00U, 0x00U};
  uint16_t raw_word;
  HAL_StatusTypeDef spi_status;

  if (raw14 == NULL)
  {
    return false;
  }

  HAL_GPIO_WritePin(EN2_CSN_GPIO_Port, EN2_CSN_Pin, GPIO_PIN_RESET);
  spi_status = HAL_SPI_TransmitReceive(&hspi1, tx_data, rx_data, 2U, MA732_SPI_TIMEOUT_MS);
  dbg_ma732_last_spi_status = (uint8_t)spi_status;
  dbg_ma732_last_spi_error = HAL_SPI_GetError(&hspi1);
  dbg_ma732_last_spi_sr = hspi1.Instance->SR;
  dbg_ma732_last_rx0 = rx_data[0];
  dbg_ma732_last_rx1 = rx_data[1];
  if (spi_status != HAL_OK)
  {
    HAL_GPIO_WritePin(EN2_CSN_GPIO_Port, EN2_CSN_Pin, GPIO_PIN_SET);
    return false;
  }
  HAL_GPIO_WritePin(EN2_CSN_GPIO_Port, EN2_CSN_Pin, GPIO_PIN_SET);

  raw_word = ((uint16_t)rx_data[0] << 8) | rx_data[1];
  dbg_ma732_last_raw_word = raw_word;
  dbg_ma732_last_raw14_msb = (raw_word >> 2) & MA732_COUNTS_MASK;
  dbg_ma732_last_raw14_lsb = raw_word & MA732_COUNTS_MASK;
  *raw14 = (raw_word >> 2) & MA732_COUNTS_MASK;
#if MA732_DIR_INVERT
  *raw14 = (uint16_t)((MA732_COUNTS_PER_TURN - *raw14) & MA732_COUNTS_MASK);
#endif
  return true;
}

static int32_t MA732_WrappedDelta(uint16_t current, uint16_t previous)
{
  int32_t delta = (int32_t)current - (int32_t)previous;

  if (delta > ((int32_t)MA732_COUNTS_PER_TURN / 2))
  {
    delta -= (int32_t)MA732_COUNTS_PER_TURN;
  }
  else if (delta < -((int32_t)MA732_COUNTS_PER_TURN / 2))
  {
    delta += (int32_t)MA732_COUNTS_PER_TURN;
  }
  else
  {
    /* no wrap correction needed */
  }

  return delta;
}

static void MA732_UpdateCache(void)
{
  uint16_t raw14 = 0U;

  if (MA732_ReadRaw14(&raw14))
  {
    __disable_irq();
    ma732_raw14_cache = raw14;
    ma732_cache_valid = 1U;
    dbg_ma732_last_read_ok = 1U;
    dbg_ma732_last_read_raw14 = raw14;
    dbg_ma732_read_ok_count++;
    __enable_irq();
  }
  else
  {
    __disable_irq();
    dbg_ma732_last_read_ok = 0U;
    dbg_ma732_read_fail_count++;
    __enable_irq();
  }
  /* On failure, keep stale cache rather than clearing it.
   * A 50µs stale angle is far better than freezing hElAngle at the last
   * value because cache_valid was cleared. TIM6 retries next tick. */
}

static bool MA732_GetCachedRaw14(uint16_t *raw14)
{
  bool valid;

  if (raw14 == NULL)
  {
    return false;
  }

  __disable_irq();
  *raw14 = ma732_raw14_cache;
  valid = (ma732_cache_valid != 0U);
  __enable_irq();

  return valid;
}

/* TIM6 update IRQ — fires at 20 kHz, refreshes the MA732 cache automatically.
 * This makes the SPI encoder self-updating like an ABI encoder timer counter:
 * ENC_CalcAngle just reads the cache without worrying about when to update it.
 * Priority 5 — below MCSDK (0–3) so FOC ISRs can preempt this if needed. */
void TIM6_DAC_IRQHandler(void)
{
  if (__HAL_TIM_GET_FLAG(&htim6, TIM_FLAG_UPDATE) &&
      __HAL_TIM_GET_IT_SOURCE(&htim6, TIM_IT_UPDATE))
  {
    __HAL_TIM_CLEAR_IT(&htim6, TIM_IT_UPDATE);
    dbg_ma732_irq_count++;
    MA732_UpdateCache();
  }
}

/* Configure TIM6 as a free-running 20 kHz interval timer.
 * 170 MHz / 170 / 50 = 20 000 Hz  (period = 49, prescaler = 169)
 * Call once after MX_SPI1_Init(), before MC_StartMotor1(). */
void MA732_TimerInit(void)
{
  __HAL_RCC_TIM6_CLK_ENABLE();

  htim6.Instance               = TIM6;
  htim6.Init.Prescaler         = 169U;   /* ÷170 → 1 MHz tick */
  htim6.Init.CounterMode       = TIM_COUNTERMODE_UP;
  htim6.Init.Period            = 49U;    /* ÷50  → 20 kHz update */
  htim6.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  HAL_TIM_Base_Init(&htim6);

  HAL_NVIC_SetPriority(TIM6_DAC_IRQn, 5, 0);
  HAL_NVIC_EnableIRQ(TIM6_DAC_IRQn);

  HAL_TIM_Base_Start_IT(&htim6);
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  /* Ensure interrupts vector to application flash (CubeMX leaves VTOR unset). */
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
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_ADC2_Init();
  MX_CORDIC_Init();
  MX_I2C3_Init();
  MX_OPAMP1_Init();
  MX_OPAMP2_Init();
  MX_TIM1_Init();
  MX_TIM4_Init();
  MX_USART1_UART_Init();
  MX_MotorControl_Init();
  MX_SPI1_Init();
  MX_FDCAN1_Init();

  /* Initialize interrupts */
  MX_NVIC_Init();
  /* USER CODE BEGIN 2 */
  HAL_GPIO_WritePin(EN2_CSN_GPIO_Port, EN2_CSN_Pin, GPIO_PIN_SET);

  /* NOTE: MX_FDCAN1_Init() is called here by CubeMX once FDCAN1 is enabled.
   * CAN_Interrupt_Init() must run after MX_FDCAN1_Init(). */
  CAN_Interrupt_Init();

  /* Start the MA732 background refresh timer — fires at 20 kHz via TIM6 IRQ.
   * Starting it here (before the 500 ms settle delay) means the cache will
   * be fully primed by the time MC_StartMotor1() calls ENC_Init(). */
  MA732_TimerInit();

  //HAL_Delay(500); /* let gate driver and sensor settle; TIM6 fills cache */

  /* Wait for the motor controller to settle into IDLE.
   * Common workflow: firmware is uploaded without battery, which immediately
   * triggers an under-voltage fault.  Keep acknowledging until the fault
   * stays gone and the state machine is truly ready.
   * Motor Pilot or a CAN CMD_MOTOR_START will perform the actual start
   * (including encoder alignment on the first run). */
  while (1)
  {
    MCI_State_t st = MC_GetSTMStateMotor1();
    if (st == FAULT_OVER)
    {
      MC_AcknowledgeFaultMotor1();        /* clear latched fault → IDLE */
    }
    else if (st == IDLE && MC_GetCurrentFaultsMotor1() == MC_NO_FAULTS)
    {
      break;                              /* ready for Motor Pilot / CAN */
    }
    HAL_GPIO_TogglePin(LED_GPIO_Port, LED_Pin);
    HAL_Delay(100);
  }

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    /* MA732 cache is kept fresh automatically by TIM6 IRQ at 20 kHz */

    /* ---- Force-realign handler ----------------------------------------
     * CMD_FORCE_ALIGN: stop motor → clear EncAligned → restart.
     * Three-phase state machine so we can wait for IDLE between steps. */
    if (force_align)
    {
      MCI_State_t st = MC_GetSTMStateMotor1();

      if (st == RUN || st == IDLE || st == FAULT_OVER || st == STOP)
      {
        if (st == RUN)
        {
          MC_StopMotor1();                        /* phase 1: request stop */
        }
        else if (st == FAULT_OVER)
        {
          MC_AcknowledgeFaultMotor1();            /* clear any fault */
        }
        else if (st == IDLE)
        {
          /* Phase 2: reset aligned flag so CHARGE_BOOT_CAP runs alignment */
          EncAlignCtrlM1.EncAligned = false;
          MC_ProgramSpeedRampMotor1(0, 0);
          MC_StartMotor1();
          force_align = 0;                        /* done */
        }
        /* if STOP, just wait — will become IDLE */
      }
    }

    {
      uint16_t raw14_snapshot;
      uint8_t cache_ok;
      __disable_irq();
      raw14_snapshot = ma732_raw14_cache;
      cache_ok = ma732_cache_valid;
      __enable_irq();

      dbg_ma732_raw14 = raw14_snapshot;
      dbg_ma732_valid = cache_ok;
      dbg_mec_angle = ENCODER_M1._Super.hMecAngle;
      dbg_el_angle = ENCODER_M1._Super.hElAngle;
      dbg_mc_state = (int16_t)MC_GetSTMStateMotor1();

      HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, (cache_ok != 0U) ? GPIO_PIN_SET : GPIO_PIN_RESET);
    }
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
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = RCC_PLLM_DIV1;
  RCC_OscInitStruct.PLL.PLLN = 10;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief NVIC Configuration.
  * @retval None
  */
static void MX_NVIC_Init(void)
{
  /* USART1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(USART1_IRQn, 3, 1);
  HAL_NVIC_EnableIRQ(USART1_IRQn);
  /* DMA1_Channel1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel1_IRQn, 3, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel1_IRQn);
  /* TIM1_BRK_TIM15_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(TIM1_BRK_TIM15_IRQn, 4, 1);
  HAL_NVIC_EnableIRQ(TIM1_BRK_TIM15_IRQn);
  /* TIM1_UP_TIM16_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(TIM1_UP_TIM16_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(TIM1_UP_TIM16_IRQn);
  /* ADC1_2_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(ADC1_2_IRQn, 2, 0);
  HAL_NVIC_EnableIRQ(ADC1_2_IRQn);
  /* TIM4_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(TIM4_IRQn, 3, 0);
  HAL_NVIC_EnableIRQ(TIM4_IRQn);
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_MultiModeTypeDef multimode = {0};
  ADC_InjectionConfTypeDef sConfigInjected = {0};
  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.DataAlign = ADC_DATAALIGN_LEFT;
  hadc1.Init.GainCompensation = 0;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc1.Init.OversamplingMode = DISABLE;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the ADC multi-mode
  */
  multimode.Mode = ADC_MODE_INDEPENDENT;
  if (HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multimode) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Injected Channel
  */
  sConfigInjected.InjectedChannel = ADC_CHANNEL_3;
  sConfigInjected.InjectedRank = ADC_INJECTED_RANK_1;
  sConfigInjected.InjectedSamplingTime = ADC_SAMPLETIME_6CYCLES_5;
  sConfigInjected.InjectedSingleDiff = ADC_SINGLE_ENDED;
  sConfigInjected.InjectedOffsetNumber = ADC_OFFSET_NONE;
  sConfigInjected.InjectedOffset = 0;
  sConfigInjected.InjectedNbrOfConversion = 1;
  sConfigInjected.InjectedDiscontinuousConvMode = DISABLE;
  sConfigInjected.AutoInjectedConv = DISABLE;
  sConfigInjected.QueueInjectedContext = DISABLE;
  sConfigInjected.ExternalTrigInjecConv = ADC_EXTERNALTRIGINJEC_T1_TRGO;
  sConfigInjected.ExternalTrigInjecConvEdge = ADC_EXTERNALTRIGINJECCONV_EDGE_RISING;
  sConfigInjected.InjecOversamplingMode = DISABLE;
  if (HAL_ADCEx_InjectedConfigChannel(&hadc1, &sConfigInjected) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_1;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_47CYCLES_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief ADC2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC2_Init(void)
{

  /* USER CODE BEGIN ADC2_Init 0 */

  /* USER CODE END ADC2_Init 0 */

  ADC_InjectionConfTypeDef sConfigInjected = {0};
  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC2_Init 1 */

  /* USER CODE END ADC2_Init 1 */

  /** Common config
  */
  hadc2.Instance = ADC2;
  hadc2.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1;
  hadc2.Init.Resolution = ADC_RESOLUTION_12B;
  hadc2.Init.DataAlign = ADC_DATAALIGN_LEFT;
  hadc2.Init.GainCompensation = 0;
  hadc2.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc2.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc2.Init.LowPowerAutoWait = DISABLE;
  hadc2.Init.ContinuousConvMode = DISABLE;
  hadc2.Init.NbrOfConversion = 1;
  hadc2.Init.DiscontinuousConvMode = DISABLE;
  hadc2.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc2.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc2.Init.DMAContinuousRequests = DISABLE;
  hadc2.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc2.Init.OversamplingMode = DISABLE;
  if (HAL_ADC_Init(&hadc2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Injected Channel
  */
  sConfigInjected.InjectedChannel = ADC_CHANNEL_3;
  sConfigInjected.InjectedRank = ADC_INJECTED_RANK_1;
  sConfigInjected.InjectedSamplingTime = ADC_SAMPLETIME_6CYCLES_5;
  sConfigInjected.InjectedSingleDiff = ADC_SINGLE_ENDED;
  sConfigInjected.InjectedOffsetNumber = ADC_OFFSET_NONE;
  sConfigInjected.InjectedOffset = 0;
  sConfigInjected.InjectedNbrOfConversion = 1;
  sConfigInjected.InjectedDiscontinuousConvMode = DISABLE;
  sConfigInjected.AutoInjectedConv = DISABLE;
  sConfigInjected.QueueInjectedContext = DISABLE;
  sConfigInjected.ExternalTrigInjecConv = ADC_EXTERNALTRIGINJEC_T1_TRGO;
  sConfigInjected.ExternalTrigInjecConvEdge = ADC_EXTERNALTRIGINJECCONV_EDGE_RISING;
  sConfigInjected.InjecOversamplingMode = DISABLE;
  if (HAL_ADCEx_InjectedConfigChannel(&hadc2, &sConfigInjected) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_5;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_47CYCLES_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  if (HAL_ADC_ConfigChannel(&hadc2, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC2_Init 2 */

  /* USER CODE END ADC2_Init 2 */

}

/**
  * @brief CORDIC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CORDIC_Init(void)
{

  /* USER CODE BEGIN CORDIC_Init 0 */

  /* USER CODE END CORDIC_Init 0 */

  /* USER CODE BEGIN CORDIC_Init 1 */

  /* USER CODE END CORDIC_Init 1 */
  hcordic.Instance = CORDIC;
  if (HAL_CORDIC_Init(&hcordic) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CORDIC_Init 2 */

  /* USER CODE END CORDIC_Init 2 */

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
  hfdcan1.Init.NominalPrescaler = 5;
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
  * @brief I2C3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C3_Init(void)
{

  /* USER CODE BEGIN I2C3_Init 0 */

  /* USER CODE END I2C3_Init 0 */

  /* USER CODE BEGIN I2C3_Init 1 */

  /* USER CODE END I2C3_Init 1 */
  hi2c3.Instance = I2C3;
  hi2c3.Init.Timing = 0x00B10E24;
  hi2c3.Init.OwnAddress1 = 0;
  hi2c3.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c3.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c3.Init.OwnAddress2 = 0;
  hi2c3.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c3.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c3.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c3) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c3, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c3, 0) != HAL_OK)
  {
    Error_Handler();
  }

  /** I2C Fast mode Plus enable
  */
  HAL_I2CEx_EnableFastModePlus(I2C_FASTMODEPLUS_I2C3);
  /* USER CODE BEGIN I2C3_Init 2 */

  /* USER CODE END I2C3_Init 2 */

}

/**
  * @brief OPAMP1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_OPAMP1_Init(void)
{

  /* USER CODE BEGIN OPAMP1_Init 0 */

  /* USER CODE END OPAMP1_Init 0 */

  /* USER CODE BEGIN OPAMP1_Init 1 */

  /* USER CODE END OPAMP1_Init 1 */
  hopamp1.Instance = OPAMP1;
  hopamp1.Init.PowerMode = OPAMP_POWERMODE_NORMALSPEED;
  hopamp1.Init.Mode = OPAMP_STANDALONE_MODE;
  hopamp1.Init.InvertingInput = OPAMP_INVERTINGINPUT_IO0;
  hopamp1.Init.NonInvertingInput = OPAMP_NONINVERTINGINPUT_IO0;
  hopamp1.Init.InternalOutput = DISABLE;
  hopamp1.Init.TimerControlledMuxmode = OPAMP_TIMERCONTROLLEDMUXMODE_DISABLE;
  hopamp1.Init.UserTrimming = OPAMP_TRIMMING_FACTORY;
  if (HAL_OPAMP_Init(&hopamp1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN OPAMP1_Init 2 */

  /* USER CODE END OPAMP1_Init 2 */

}

/**
  * @brief OPAMP2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_OPAMP2_Init(void)
{

  /* USER CODE BEGIN OPAMP2_Init 0 */

  /* USER CODE END OPAMP2_Init 0 */

  /* USER CODE BEGIN OPAMP2_Init 1 */

  /* USER CODE END OPAMP2_Init 1 */
  hopamp2.Instance = OPAMP2;
  hopamp2.Init.PowerMode = OPAMP_POWERMODE_NORMALSPEED;
  hopamp2.Init.Mode = OPAMP_STANDALONE_MODE;
  hopamp2.Init.InvertingInput = OPAMP_INVERTINGINPUT_IO1;
  hopamp2.Init.NonInvertingInput = OPAMP_NONINVERTINGINPUT_IO2;
  hopamp2.Init.InternalOutput = DISABLE;
  hopamp2.Init.TimerControlledMuxmode = OPAMP_TIMERCONTROLLEDMUXMODE_DISABLE;
  hopamp2.Init.UserTrimming = OPAMP_TRIMMING_FACTORY;
  if (HAL_OPAMP_Init(&hopamp2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN OPAMP2_Init 2 */

  /* USER CODE END OPAMP2_Init 2 */

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
  hspi1.Init.DataSize = SPI_DATASIZE_4BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_HIGH;
  hspi1.Init.CLKPhase = SPI_PHASE_2EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_64;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 7;
  hspi1.Init.CRCLength = SPI_CRC_LENGTH_DATASIZE;
  hspi1.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */
  /* CubeMX was left at 4-bit data frames (only 8 clocks for Size=2).
   * MA732 needs 16 clock cycles for a full 16-bit angle word.
   * Re-init with 8-bit frames so Size=2 generates the required 16 clocks. */
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  if (HAL_SPI_Init(&hspi1) != HAL_OK) { Error_Handler(); }
  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = ((TIM_CLOCK_DIVIDER) - 1);
  htim1.Init.CounterMode = TIM_COUNTERMODE_CENTERALIGNED1;
  htim1.Init.Period = ((PWM_PERIOD_CYCLES) / 2);
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV2;
  htim1.Init.RepetitionCounter = (REP_COUNTER);
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_OC4REF;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = ((PWM_PERIOD_CYCLES) / 4);
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM2;
  sConfigOC.Pulse = (((PWM_PERIOD_CYCLES) / 2) - (HTMIN));
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_ENABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_ENABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = ((DEAD_TIME_COUNTS) / 2);
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.BreakFilter = 0;
  sBreakDeadTimeConfig.BreakAFMode = TIM_BREAK_AFMODE_INPUT;
  sBreakDeadTimeConfig.Break2State = TIM_BREAK2_DISABLE;
  sBreakDeadTimeConfig.Break2Polarity = TIM_BREAK2POLARITY_HIGH;
  sBreakDeadTimeConfig.Break2Filter = 3;
  sBreakDeadTimeConfig.Break2AFMode = TIM_BREAK_AFMODE_INPUT;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */
  HAL_TIM_MspPostInit(&htim1);

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 0;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = M1_PULSE_NBR;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = M1_ENC_IC_FILTER;
  sConfig.IC2Polarity = TIM_ICPOLARITY_FALLING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = M1_ENC_IC_FILTER;
  if (HAL_TIM_Encoder_Init(&htim4, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */

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
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMAMUX1_CLK_ENABLE();
  __HAL_RCC_DMA1_CLK_ENABLE();

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
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(EN2_CSN_GPIO_Port, EN2_CSN_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GD_WAKE_GPIO_Port, GD_WAKE_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : EN2_CSN_Pin */
  GPIO_InitStruct.Pin = EN2_CSN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(EN2_CSN_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : GD_WAKE_Pin */
  GPIO_InitStruct.Pin = GD_WAKE_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GD_WAKE_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : GD_READY_Pin GD_NFAULT_Pin */
  GPIO_InitStruct.Pin = GD_READY_Pin|GD_NFAULT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pin : LED_Pin */
  GPIO_InitStruct.Pin = LED_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LED_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
void ENC_Init(ENCODER_Handle_t *pHandle)
{
  uint16_t raw14;
  uint8_t index;

  if (pHandle == NULL)
  {
    return;
  }

  for (index = 0U; index < pHandle->SpeedBufferSize; index++)
  {
    pHandle->DeltaCapturesBuffer[index] = 0;
  }
  pHandle->U32MAXdivPulseNumber = UINT32_MAX / MA732_COUNTS_PER_TURN;
  pHandle->SpeedSamplingFreqUnit = ((uint32_t)pHandle->SpeedSamplingFreqHz * (uint32_t)SPEED_UNIT);
  pHandle->DeltaCapturesIndex = 0U;
  pHandle->TimerOverflowNb = 0U;
  pHandle->TimerOverflowError = false;
  pHandle->SensorIsReliable = true;

  if (MA732_GetCachedRaw14(&raw14))
  {
    pHandle->PreviousCapture = (uint32_t)raw14;
    ma732_last_raw14_for_pos = raw14;
    ma732_elec_accum_counts = 0;
    ma732_pos_initialized = 1U;
  }
  else
  {
    pHandle->PreviousCapture = 0U;
    ma732_last_raw14_for_pos = 0U;
    ma732_elec_accum_counts = 0;
    ma732_pos_initialized = 0U;
    pHandle->SensorIsReliable = false;
  }
}

void ENC_Clear(ENCODER_Handle_t *pHandle)
{
  uint16_t raw14;
  uint8_t index;

  if (pHandle == NULL)
  {
    return;
  }

  for (index = 0U; index < pHandle->SpeedBufferSize; index++)
  {
    pHandle->DeltaCapturesBuffer[index] = 0;
  }
  pHandle->DeltaCapturesIndex = 0U;
  pHandle->_Super.hAvrMecSpeedUnit = 0;
  pHandle->_Super.hMecAccelUnitP = 0;
  pHandle->_Super.hElSpeedDpp = 0;
  pHandle->SensorIsReliable = true;

  if (MA732_GetCachedRaw14(&raw14))
  {
    pHandle->PreviousCapture = (uint32_t)raw14;
    ma732_last_raw14_for_pos = raw14;
    ma732_elec_accum_counts =
      ((int32_t)(uint16_t)pHandle->_Super.hMecAngle * (int32_t)MA732_COUNTS_PER_TURN *
       (int32_t)pHandle->_Super.bElToMecRatio) / 65536;
    ma732_pos_initialized = 1U;
  }
  else
  {
    pHandle->PreviousCapture = 0U;
    ma732_last_raw14_for_pos = 0U;
    ma732_pos_initialized = 0U;
  }
}

int16_t ENC_CalcAngle(ENCODER_Handle_t *pHandle)
{
  uint16_t raw14;
  uint32_t pp;
  int32_t elec_counts;
  int32_t mech_divisor_counts;
  int32_t delta_counts;
  int32_t mech_mod_counts;
  int16_t mec_angle;
  int16_t el_angle;
  int32_t elec_offset;

  if (pHandle == NULL)
  {
    return 0;
  }

  if (!MA732_GetCachedRaw14(&raw14))
  {
    return pHandle->_Super.hElAngle;
  }

  pp = (uint32_t)pHandle->_Super.bElToMecRatio;
  if (pp == 0U)
  {
    pp = 1U;
  }

  if (ma732_pos_initialized == 0U)
  {
    ma732_last_raw14_for_pos = raw14;
    ma732_elec_accum_counts = 0;
    ma732_pos_initialized = 1U;
  }

  delta_counts = MA732_WrappedDelta(raw14, ma732_last_raw14_for_pos);
  ma732_elec_accum_counts += delta_counts;
  ma732_last_raw14_for_pos = raw14;

  __disable_irq();
  elec_offset = ma732_elec_offset_counts;
  __enable_irq();
  elec_counts = ((int32_t)raw14 + elec_offset) % (int32_t)MA732_COUNTS_PER_TURN;
  if (elec_counts < 0)
  {
    elec_counts += (int32_t)MA732_COUNTS_PER_TURN;
  }
  el_angle = (int16_t)((uint32_t)elec_counts * MA732_DPP_PER_COUNT);

  mech_divisor_counts = (int32_t)MA732_COUNTS_PER_TURN * (int32_t)pp;
  mech_mod_counts = ma732_elec_accum_counts % mech_divisor_counts;
  if (mech_mod_counts < 0)
  {
    mech_mod_counts += mech_divisor_counts;
  }

  mec_angle = (int16_t)(((uint32_t)mech_mod_counts * 65536U) / (uint32_t)mech_divisor_counts);
  pHandle->_Super.hMecAngle = mec_angle;
  pHandle->_Super.wMecAngle +=
    (delta_counts * (int32_t)MA732_DPP_PER_COUNT) / (int32_t)pp;

  pHandle->_Super.hElAngle = el_angle;

  return el_angle;
}

bool ENC_CalcAvrgMecSpeedUnit(ENCODER_Handle_t *pHandle, int16_t *pMecSpeedUnit)
{
  uint16_t raw14;
  uint32_t pp;
  int32_t delta_counts;
  int32_t speed_sum = 0;
  int32_t speed_unit;
  int64_t inst_el_speed_dpp;
  uint8_t index;

  if ((pHandle == NULL) || (pMecSpeedUnit == NULL))
  {
    return false;
  }

  if (!MA732_GetCachedRaw14(&raw14))
  {
    pHandle->SensorIsReliable = false;
    pHandle->_Super.bSpeedErrorNumber = pHandle->_Super.bMaximumSpeedErrorsNumber;
    *pMecSpeedUnit = 0;
    return false;
  }

  pp = (uint32_t)pHandle->_Super.bElToMecRatio;
  if (pp == 0U)
  {
    pp = 1U;
  }

  delta_counts = MA732_WrappedDelta(raw14,
                                    (uint16_t)(pHandle->PreviousCapture & MA732_COUNTS_MASK));

  pHandle->DeltaCapturesBuffer[pHandle->DeltaCapturesIndex] = delta_counts;
  pHandle->DeltaCapturesIndex++;
  if (pHandle->DeltaCapturesIndex >= pHandle->SpeedBufferSize)
  {
    pHandle->DeltaCapturesIndex = 0U;
  }

  for (index = 0U; index < pHandle->SpeedBufferSize; index++)
  {
    speed_sum += pHandle->DeltaCapturesBuffer[index];
  }

  speed_unit = (speed_sum * (int32_t)pHandle->SpeedSamplingFreqUnit) /
               ((int32_t)MA732_COUNTS_PER_TURN * (int32_t)pHandle->SpeedBufferSize * (int32_t)pp);
  *pMecSpeedUnit = (int16_t)speed_unit;

  pHandle->_Super.hMecAccelUnitP = (int16_t)(speed_unit - pHandle->_Super.hAvrMecSpeedUnit);
  pHandle->_Super.hAvrMecSpeedUnit = (int16_t)speed_unit;

  inst_el_speed_dpp = (int64_t)delta_counts * (int64_t)pHandle->SpeedSamplingFreqHz *
                      (int64_t)pHandle->_Super.DPPConvFactor;
  inst_el_speed_dpp /= ((int64_t)MA732_COUNTS_PER_TURN * (int64_t)pHandle->_Super.hMeasurementFrequency);
  pHandle->_Super.hElSpeedDpp = (int16_t)inst_el_speed_dpp;

  pHandle->PreviousCapture = (uint32_t)raw14;
  pHandle->SensorIsReliable = SPD_IsMecSpeedReliable(&pHandle->_Super, pMecSpeedUnit);

  return pHandle->SensorIsReliable;
}

void ENC_SetMecAngle(ENCODER_Handle_t *pHandle, int16_t hMecAngle)
{
  uint16_t raw14_now;
  uint32_t pp;
  uint32_t target_mec_counts;
  uint16_t target_el_counts;
  uint16_t current_el_counts;
  int32_t new_offset;

  if (pHandle == NULL)
  {
    return;
  }

  pp = (uint32_t)pHandle->_Super.bElToMecRatio;
  if (pp == 0U)
  {
    pp = 1U;
  }

  /* Use the cache (kept fresh at 20 kHz by TIM6) instead of a direct SPI
   * read. ENC_SetMecAngle is called from TIMx_BRK ISR (priority 4) which
   * can preempt TIM6 (priority 5) mid-SPI — a direct read would get
   * HAL_BUSY and silently bail, breaking alignment every call. */
  if (!MA732_GetCachedRaw14(&raw14_now))
  {
    return;
  }

  target_mec_counts =
    ((uint32_t)(uint16_t)hMecAngle * (uint32_t)MA732_COUNTS_PER_TURN) / 65536U;
  target_el_counts = (uint16_t)((target_mec_counts * pp) & MA732_COUNTS_MASK);
  current_el_counts = raw14_now;

  new_offset = (int32_t)target_el_counts - (int32_t)current_el_counts;
  new_offset %= (int32_t)MA732_COUNTS_PER_TURN;
  if (new_offset < 0)
  {
    new_offset += (int32_t)MA732_COUNTS_PER_TURN;
  }

  __disable_irq();
  ma732_elec_offset_counts = new_offset;
  ma732_raw14_cache = raw14_now;
  ma732_cache_valid = 1U;
  ma732_last_raw14_for_pos = raw14_now;
  ma732_pos_initialized = 1U;
  ma732_elec_accum_counts = (int32_t)target_el_counts;
  pHandle->PreviousCapture = (uint32_t)raw14_now;
  pHandle->_Super.hMecAngle = hMecAngle;
  pHandle->_Super.hElAngle = (int16_t)(hMecAngle * (int16_t)pp);
  __enable_irq();
}

/* --------------------------------------------------------------------------
 * Impedance controller — runs at 1 kHz via the MCSDK medium-frequency hook.
 * τ = kp*(p_ref - p_actual) + kd*(v_ref - v_actual) + τ_ff
 * -------------------------------------------------------------------------- */
void MC_APP_PostMediumFrequencyHook_M1(void)
{
    if (MC_GetSTMStateMotor1() != RUN)
        return;
    if (!can_active)
        return;  /* don't fight motor profiler before first CAN command */

    float pos_actual = (float)ENCODER_M1._Super.hMecAngle * S16_TO_RADIANS;
    float vel_actual = ENCODER_VEL_SIGN * MC_GetAverageMecSpeedMotor1_F() * RPM_TO_RADS;

    float tau = ImpedanceCtrl.kp * (ImpedanceCtrl.position_target - pos_actual)
              + ImpedanceCtrl.kd * (ImpedanceCtrl.velocity_target - vel_actual)
              + ImpedanceCtrl.torque_target;

    if (tau >  TORQUE_MAX_NM) tau =  TORQUE_MAX_NM;
    if (tau < -TORQUE_MAX_NM) tau = -TORQUE_MAX_NM;

    /* Clamp Iq in Amps before passing to MCSDK to prevent int16 overflow
     * in MCI_ExecTorqueRamp_F (IQMAX_A = 43 A hardware limit). */
    float iq_cmd = tau / MOTOR_KT;
    if (iq_cmd >  IQ_MAX_A) iq_cmd =  IQ_MAX_A;
    if (iq_cmd < -IQ_MAX_A) iq_cmd = -IQ_MAX_A;

    MC_ProgramTorqueRampMotor1_F(iq_cmd, 0);
}

/* ---- CAN packing helpers ------------------------------------------------ */

float Math_Map_Int_To_Float(int x_int, float x_min, float x_max, int bits)
{
    float total = (float)((1 << bits) - 1);
    return x_min + ((float)x_int * (x_max - x_min) / total);
}

int Math_Map_Float_To_Int(float x, float x_min, float x_max, int bits)
{
    if (x < x_min) x = x_min;
    if (x > x_max) x = x_max;
    float total = (float)((1 << bits) - 1);
    return (int)((x - x_min) * total / (x_max - x_min));
}

void CAN_Unpack_Command(const uint8_t *data, CAN_Command_t *cmd)
{
    const float P_MIN = -(float)M_PI,  P_MAX = (float)M_PI;
    const float V_MIN = -1500.0f,      V_MAX = 1500.0f;
    const float KP_MIN = 0.0f,         KP_MAX = 500.0f;
    const float KD_MIN = 0.0f,         KD_MAX = 15.0f;
    const float TRQ_MIN = -TORQUE_MAX_NM, TRQ_MAX = TORQUE_MAX_NM;

    uint16_t p_int  = ((uint16_t)data[0] << 8) | data[1];
    uint16_t v_int  = ((uint16_t)data[2] << 4) | (data[3] >> 4);
    uint16_t kp_int = ((uint16_t)(data[3] & 0x0F) << 8) | data[4];
    uint16_t kd_int = ((uint16_t)data[5] << 4) | (data[6] >> 4);
    uint16_t t_int  = ((uint16_t)(data[6] & 0x0F) << 8) | data[7];

    cmd->position_target = Math_Map_Int_To_Float(p_int,  P_MIN,  P_MAX,  16);
    cmd->velocity_target = Math_Map_Int_To_Float(v_int,  V_MIN,  V_MAX,  12);
    cmd->kp              = Math_Map_Int_To_Float(kp_int, KP_MIN, KP_MAX, 12);
    cmd->kd              = Math_Map_Int_To_Float(kd_int, KD_MIN, KD_MAX, 12);
    cmd->torque_ff       = Math_Map_Int_To_Float(t_int,  TRQ_MIN, TRQ_MAX, 12);
}

void CAN_Pack_Feedback(uint8_t *data, const CAN_Feedback_t *fb)
{
    const float P_MIN = -(float)M_PI,   P_MAX = (float)M_PI;
    const float V_MIN = -1500.0f,       V_MAX = 1500.0f;
    const float TRQ_MIN = -TORQUE_MAX_NM, TRQ_MAX = TORQUE_MAX_NM;

    uint16_t p_int = (uint16_t)Math_Map_Float_To_Int(fb->position_actual, P_MIN, P_MAX, 16);
    uint16_t v_int = (uint16_t)Math_Map_Float_To_Int(fb->velocity_actual, V_MIN, V_MAX, 16);
    uint16_t t_int = (uint16_t)Math_Map_Float_To_Int(fb->torque_actual,   TRQ_MIN, TRQ_MAX, 16);

    MCI_State_t st = MC_GetSTMStateMotor1();
    uint8_t state_nibble;
    switch (st) {
        case IDLE:       state_nibble = 0; break;
        case RUN:        state_nibble = 1; break;
        case STOP:       state_nibble = 2; break;
        case FAULT_NOW:  state_nibble = 3; break;
        case FAULT_OVER: state_nibble = 4; break;
        default:         state_nibble = 5; break;
    }
    data[0] = (uint8_t)((state_nibble << 4) | (fb->motor_id & 0x0Fu));
    data[1] = fb->error_code;
    data[2] = (uint8_t)((p_int >> 8) & 0xFFu);
    data[3] = (uint8_t)(p_int & 0xFFu);
    data[4] = (uint8_t)((v_int >> 8) & 0xFFu);
    data[5] = (uint8_t)(v_int & 0xFFu);
    data[6] = (uint8_t)((t_int >> 8) & 0xFFu);
    data[7] = (uint8_t)(t_int & 0xFFu);
}

void CAN_Interrupt_Init(void)
{
    FDCAN_FilterTypeDef filterConfig;
    filterConfig.IdType       = FDCAN_STANDARD_ID;
    filterConfig.FilterIndex  = 0;
    filterConfig.FilterType   = FDCAN_FILTER_RANGE;
    filterConfig.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
    filterConfig.FilterID1    = 0x000;
    filterConfig.FilterID2    = 0x7FF;
    if (HAL_FDCAN_ConfigFilter(&hfdcan1, &filterConfig) != HAL_OK)
        Error_Handler();

    if (HAL_FDCAN_Start(&hfdcan1) != HAL_OK)
        Error_Handler();

    if (HAL_FDCAN_ActivateNotification(&hfdcan1, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0) != HAL_OK)
        Error_Handler();
}

void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t RxFifo0ITs)
{
    uint8_t should_tx_feedback = 0U;

    if ((RxFifo0ITs & FDCAN_IT_RX_FIFO0_NEW_MESSAGE) == 0)
        return;

    FDCAN_RxHeaderTypeDef rxHeader;
    uint8_t rxData[8];
    if (HAL_FDCAN_GetRxMessage(hfdcan, FDCAN_RX_FIFO0, &rxHeader, rxData) != HAL_OK)
        return;

    can_rx_count++;

    if (rxHeader.Identifier == MY_MOTOR_ID)
    {
        should_tx_feedback = 1U;

        if (MC_GetSTMStateMotor1() == FAULT_OVER)
        {
            ImpedanceCtrl.torque_target   = 0.0f;
            ImpedanceCtrl.position_target = ImpedanceCtrl.position_curr;
            ImpedanceCtrl.velocity_target = 0.0f;
            ImpedanceCtrl.kp = 0.0f;
            ImpedanceCtrl.kd = 0.0f;
            MC_AcknowledgeFaultMotor1();
        }

        CAN_Unpack_Command(rxData, &IncomingCmd);
        can_active = 1; /* first valid command — enable the impedance controller hook */

        __disable_irq();
        ImpedanceCtrl.position_target = IncomingCmd.position_target;
        ImpedanceCtrl.velocity_target = IncomingCmd.velocity_target;
        ImpedanceCtrl.torque_target   = IncomingCmd.torque_ff;
        ImpedanceCtrl.kp              = IncomingCmd.kp;
        ImpedanceCtrl.kd              = IncomingCmd.kd;
        ImpedanceCtrl.steps_per_can_frame   = 20;
        ImpedanceCtrl.position_step = (ImpedanceCtrl.position_target - ImpedanceCtrl.position_curr) / 20.0f;
        ImpedanceCtrl.velocity_step = (ImpedanceCtrl.velocity_target - ImpedanceCtrl.velocity_curr) / 20.0f;
        ImpedanceCtrl.torque_step   = (ImpedanceCtrl.torque_target   - ImpedanceCtrl.torque_curr)   / 20.0f;
        ImpedanceCtrl.interpolation_counter = 0;
        __enable_irq();
    }
    else if (rxHeader.Identifier == CTRL_ID
             && rxHeader.RxFrameType == FDCAN_DATA_FRAME
             && rxHeader.DataLength  == FDCAN_DLC_BYTES_1)
    {
        should_tx_feedback = 1U;

        if (rxData[0] == CMD_MOTOR_START)
        {
            MC_AcknowledgeFaultMotor1();
            ImpedanceCtrl.torque_target   = 0.0f;
            ImpedanceCtrl.position_target = ImpedanceCtrl.position_curr;
            ImpedanceCtrl.velocity_target = 0.0f;
            ImpedanceCtrl.kp = 0.0f;
            ImpedanceCtrl.kd = 0.0f;
            /* Prime a valid buffered command (required before MC_StartMotor1).
             * Speed mode at 0 RPM keeps the motor still until the first CAN
             * data frame arrives and the impedance hook switches to torque mode. */
            MC_ProgramSpeedRampMotor1(0, 0);
            MC_StartMotor1();
        }
        else if (rxData[0] == CMD_MOTOR_STOP)
        {
            MC_StopMotor1();
        }
        else if (rxData[0] == CMD_FORCE_ALIGN)
        {
            /* Signal the main loop to stop, clear alignment, and restart.
             * Can't call MC_StopMotor1 + MC_StartMotor1 back-to-back here
             * because we need to wait for IDLE between them. */
            force_align = 1;
        }
        else if (rxData[0] == CMD_FEEDBACK_REQUEST)
        {
            /* Feedback is packed below without changing motor state. */
        }
        else
        {
            should_tx_feedback = 0U;
        }
    }

    if (should_tx_feedback != 0U)
    {
        CurrentState.motor_id       = MY_MOTOR_ID;
        CurrentState.error_code     = (uint8_t)(MC_GetOccurredFaultsMotor1() & 0xFFu);
        CurrentState.position_actual = (float)ENCODER_M1._Super.hMecAngle * S16_TO_RADIANS;
        CurrentState.velocity_actual = MC_GetAverageMecSpeedMotor1_F() * RPM_TO_RADS;
        qd_f_t iqd = MC_GetIqdMotor1_F();
        CurrentState.torque_actual  = iqd.q * MOTOR_KT;

        FDCAN_TxHeaderTypeDef txHeader;
        uint8_t txData[8];
        CAN_Pack_Feedback(txData, &CurrentState);
        txHeader.Identifier            = MY_MOTOR_ID + 0x100;
        txHeader.IdType                = FDCAN_STANDARD_ID;
        txHeader.TxFrameType           = FDCAN_DATA_FRAME;
        txHeader.DataLength            = FDCAN_DLC_BYTES_8;
        txHeader.ErrorStateIndicator   = FDCAN_ESI_ACTIVE;
        txHeader.BitRateSwitch         = FDCAN_BRS_OFF;
        txHeader.FDFormat              = FDCAN_CLASSIC_CAN;
        txHeader.TxEventFifoControl    = FDCAN_NO_TX_EVENTS;
        txHeader.MessageMarker         = 0;
        HAL_FDCAN_AddMessageToTxFifoQ(hfdcan, &txHeader, txData);
    }
}

int _write(int fd, char *ptr, int len)
{
    if (fd == 1 || fd == 2)
    {
        HAL_StatusTypeDef s = HAL_UART_Transmit(&huart1, (uint8_t *)ptr, (uint16_t)len, 100);
        return (s == HAL_OK) ? len : -1;
    }
    return -1;
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
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
