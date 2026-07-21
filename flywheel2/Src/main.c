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
/* Header includes required for MC definitions */
#include "mc_api.h"
#include "mc_config.h"
#include "mc_type.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct {
    float position_target; // Rad
    float velocity_target; // Rad/s
    float kp;              // Nm/Rad
    float kd;              // Nm/(Rad/s)
    float torque_ff;       // Nm
} CAN_Command_t;

typedef struct {
    float position_actual;
    float velocity_actual;
    float torque_actual;
    uint8_t motor_id;
    uint8_t error_code;
} CAN_Feedback_t;

// Definition for the real-time tracking interpolation structure
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
#define MOTOR_KT          0.0095f     // [Nm/A] - torque constant
//#define IQMAX_A           20.0f       // [A]  - must match IQMAX_A in drive_parameters.h
#define TORQUE_MAX_NM     0.5f        // [Nm] - peak torque headroom (> IQMAX_A*KT = 0.19 Nm steady-state)

/* TEMPORARY bench test: set to 0 before restoring normal CAN torque control. */
#define TEMP_TORQUE_RAMP_TEST_ENABLE  0U
#define TEMP_TORQUE_TEST_NM           0.02f
#define TEMP_TORQUE_RAMP_MS           1000U
#define TEMP_TORQUE_HOLD_MS           2000U

/* Encoder velocity sign: set to +1.0f or -1.0f.
 * Test: spin motor by hand in positive direction (same as commanded).
 * Check velocity feedback in CAN response:
 *   - If positive  → keep  +1.0f
 *   - If negative  → change to -1.0f  (kd will runaway without this)
 * Tuning note: with pos_actual in radians (-π..+π), use kp in range 50–500 Nm/rad. */
#define ENCODER_VEL_SIGN  1.0f

// Conversion constants
#define DPP_TO_RADIANS (2.0f * 3.14159265f / 65536.0f)
#define RPM_TO_RADS    0.104719755f
#define M_PI  3.1415926

// Conversion constant: ST SDK electrical angle is usually an s16 mapping from -32768 to 32767
#define S16_TO_RADIANS    (3.14159265f / 32768.0f)
// 65536 total steps across 2*PI radians
#define DPP_TO_RADIANS (2.0f * 3.14159265f / 65536.0f)
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

TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim4;

UART_HandleTypeDef huart1;
DMA_HandleTypeDef hdma_usart1_rx;
DMA_HandleTypeDef hdma_usart1_tx;

/* USER CODE BEGIN PV */
CAN_Command_t         IncomingCmd  = {0};
CAN_Feedback_t        CurrentState = {0};
ImpedanceController_t ImpedanceCtrl = {0};

volatile uint32_t can_rx_count = 0;
volatile uint8_t  force_align  = 0; /* set by CAN CMD_FORCE_ALIGN; handled in main loop */

/* Motor ID detected at startup from IDENTITY pin (PC13):
 *   PC13 floating (pulled high) → ID 0x01
 *   PC13 shorted to GND         → ID 0x02
 * g_ctrl_id is the state-machine command address = motor_id + 0x200 */
uint8_t  g_motor_id = 0x01;
uint32_t g_ctrl_id  = 0x201;

// Control command bytes (1-byte payload on CTRL_ID)
#define CMD_MOTOR_START  0x01
#define CMD_MOTOR_STOP   0x02
#define CMD_FEEDBACK_REQUEST 0x04
#define CMD_FORCE_ALIGN  0x03  /* stop → clear encoder alignment → restart */

// Motor parameters
#define MOTOR_POLE_PAIRS  POLE_PAIR_NUM  // [Pole Pairs] - pulled from pmsm_motor_parameters.h

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
static void MX_FDCAN1_Init(void);
static void MX_NVIC_Init(void);
/* USER CODE BEGIN PFP */
void CAN_Interrupt_Init(void);
float Math_Map_Int_To_Float(int x_int, float x_min, float x_max, int bits);
int Math_Map_Float_To_Int(float x, float x_min, float x_max, int bits);
void CAN_Unpack_Command(const uint8_t* data, CAN_Command_t* cmd);
void CAN_Pack_Feedback(uint8_t* data, const CAN_Feedback_t* fb);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

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
  MX_FDCAN1_Init();

  /* Initialize interrupts */
  MX_NVIC_Init();
  /* USER CODE BEGIN 2 */

  /* Read IDENTITY strap (PC13) to determine motor ID.
   * MX configures PC13 as input with internal pull-up.
   * Floating → reads HIGH → ID 0x01
   * Shorted to GND → reads LOW → ID 0x02 */
  if (HAL_GPIO_ReadPin(IDENTITY_GPIO_Port, IDENTITY_Pin) == GPIO_PIN_RESET)
    g_motor_id = 0x02;
  else
    g_motor_id = 0x01;
  g_ctrl_id = g_motor_id + 0x200;

  CAN_Interrupt_Init();

    HAL_Delay(500);//let things start
    MC_AcknowledgeFaultMotor1();
    MC_ProgramTorqueRampMotor1(0,0);
    MC_StartMotor1();
    while(MC_GetSTMStateMotor1() != RUN){
  	  HAL_GPIO_TogglePin(LED_GPIO_Port, LED_Pin);
  	  HAL_Delay(200);
    }
    //  while(MC_GetAlignmentStatusMotor1() != TC_ALIGNMENT_COMPLETED){
    //	  //wait
    //  } for pos control only

//    MC_StartMotor1();
//    HAL_Delay(500);
//    MC_AcknowledgeFaultMotor1();
//    MC_StopMotor1();
//    HAL_Delay(500);
//    MC_StartMotor1();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
#if (TEMP_TORQUE_RAMP_TEST_ENABLE == 1U)
    static uint8_t torque_test_phase = 0U;
    static uint32_t torque_test_phase_start_ms = 0U;
    uint32_t torque_test_now_ms = HAL_GetTick();

    if (MC_GetSTMStateMotor1() != RUN)
    {
      torque_test_phase = 0U;
    }
    else if (torque_test_phase == 0U)
    {
      MC_ProgramTorqueRampMotor1_F(TEMP_TORQUE_TEST_NM / MOTOR_KT,
                                   TEMP_TORQUE_RAMP_MS);
      HAL_GPIO_TogglePin(LED_GPIO_Port, LED_Pin);
      torque_test_phase_start_ms = torque_test_now_ms;
      torque_test_phase = 1U;
    }
    else if ((uint32_t)(torque_test_now_ms - torque_test_phase_start_ms) >=
             (TEMP_TORQUE_RAMP_MS + TEMP_TORQUE_HOLD_MS))
    {
      float torque_nm = (torque_test_phase == 1U) ?
                        -TEMP_TORQUE_TEST_NM : TEMP_TORQUE_TEST_NM;
      MC_ProgramTorqueRampMotor1_F(torque_nm / MOTOR_KT,
                                   TEMP_TORQUE_RAMP_MS);
      HAL_GPIO_TogglePin(LED_GPIO_Port, LED_Pin);
      torque_test_phase_start_ms = torque_test_now_ms;
      torque_test_phase = (torque_test_phase == 1U) ? 2U : 1U;
    }
#endif

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
          MC_ProgramTorqueRampMotor1(0, 0);
          MC_StartMotor1();
          force_align = 0;                        /* done */
        }
        /* if STOP, just wait — will become IDLE */
      }
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
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GD_WAKE_GPIO_Port, GD_WAKE_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : IDENTITY_Pin */
  GPIO_InitStruct.Pin = IDENTITY_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(IDENTITY_GPIO_Port, &GPIO_InitStruct);

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

/**
 * @brief Override the MCSDK medium-frequency hook (runs at 1 kHz) to apply
 *        the full impedance-control torque command to the motor.
 *        τ = kp*(p_ref - p) + kd*(v_ref - v) + τ_ff
 */
void MC_APP_PostMediumFrequencyHook_M1(void)
{
#if (TEMP_TORQUE_RAMP_TEST_ENABLE == 1U)
    return;
#endif

    /* Only run the impedance controller when the motor is actually spinning.
     * In FAULT_NOW / FAULT_OVER / IDLE etc. skip the torque command so that
     * stale targets don't get replayed the moment the motor is restarted. */
    if (MC_GetSTMStateMotor1() != RUN)
        return;

    /* 1. Current mechanical position (rad), converted from s16degrees format
     * s16degrees: -32768 = -π rad, +32767 = +π rad (wraps each revolution) */
    float pos_actual = (float)ENCODER_M1._Super.hMecAngle * ((float)M_PI / 32768.0f);

    /* 2. Current mechanical velocity (rad/s) — sign must match positive direction convention */
    float vel_actual = ENCODER_VEL_SIGN * MC_GetAverageMecSpeedMotor1_F() * RPM_TO_RADS;

    /* 3. Full impedance torque (Nm) */
    float tau = ImpedanceCtrl.kp * (ImpedanceCtrl.position_target - pos_actual)
              + ImpedanceCtrl.kd * (ImpedanceCtrl.velocity_target - vel_actual)
              + ImpedanceCtrl.torque_target;

    /* 4. Clamp to physical torque limit (IQMAX_A × KT = 0.19 Nm) */
    if (tau >  TORQUE_MAX_NM) tau =  TORQUE_MAX_NM;
    if (tau < -TORQUE_MAX_NM) tau = -TORQUE_MAX_NM;

    /* 5. Convert Nm -> Iq (A) and apply immediately (duration = 0 ms) */
    MC_ProgramTorqueRampMotor1_F(tau / MOTOR_KT, 0);
}

void CAN_Unpack_Command(const uint8_t* data, CAN_Command_t* cmd) {
    // Define the exact hardcoded boundaries used by your main controller
    const float P_MIN = -(float)M_PI;  const float P_MAX = (float)M_PI;
    const float V_MIN = -45.0f;  const float V_MAX = 45.0f;
    const float KP_MIN = 0.0f;   const float KP_MAX = 500.0f;
    const float KD_MIN = 0.0f;   const float KD_MAX = 15.0f;
    const float Torque_MIN = -TORQUE_MAX_NM;  const float Torque_MAX = TORQUE_MAX_NM;

    // 1. Reconstruct integers from the byte stream
    uint16_t p_int  = (data[0] << 8) | data[1];
    uint16_t v_int  = (data[2] << 4) | (data[3] >> 4);
    uint16_t kp_int = ((data[3] & 0x0F) << 8) | data[4];
    uint16_t kd_int = (data[5] << 4) | (data[6] >> 4);
    uint16_t t_int  = ((data[6] & 0x0F) << 8) | data[7];

    // 2. Convert integers back to physical float values
    cmd->position_target = Math_Map_Int_To_Float(p_int,  P_MIN,  P_MAX,  16);
    cmd->velocity_target = Math_Map_Int_To_Float(v_int,  V_MIN,  V_MAX,  12);
    cmd->kp              = Math_Map_Int_To_Float(kp_int, KP_MIN, KP_MAX, 12);
    cmd->kd              = Math_Map_Int_To_Float(kd_int, KD_MIN, KD_MAX, 12);
    cmd->torque_ff       = Math_Map_Int_To_Float(t_int,  Torque_MIN,  Torque_MAX,  12);
}

// Helper function to map floats to integers
int Math_Map_Float_To_Int(float x, float x_min, float x_max, int bits) {
    if (x < x_min) x = x_min;
    if (x > x_max) x = x_max;
    float total_intervals = (float)((1 << bits) - 1);
    return (int)((x - x_min) * total_intervals / (x_max - x_min));
}
float Math_Map_Int_To_Float(int x_int, float x_min, float x_max, int bits) {
    float total_intervals = (float)((1 << bits) - 1);
    return x_min + ((float)x_int * (x_max - x_min) / total_intervals);
}

void CAN_Pack_Feedback(uint8_t* data, const CAN_Feedback_t* fb) {
    const float P_MIN = -(float)M_PI; const float P_MAX = (float)M_PI;
    const float V_MIN = -1500.0f; const float V_MAX = 1500.0f;
    const float Torque_MIN = -TORQUE_MAX_NM; const float Torque_MAX = TORQUE_MAX_NM;

    // Convert actual variables to integers
    uint16_t p_int = Math_Map_Float_To_Int(fb->position_actual, P_MIN, P_MAX, 16);
    uint16_t v_int = Math_Map_Float_To_Int(fb->velocity_actual, V_MIN, V_MAX, 16);
    uint16_t t_int = Math_Map_Float_To_Int(fb->torque_actual,   Torque_MIN, Torque_MAX, 16);

    // Stuff integers into the 8-byte array
    // data[0]: upper nibble = compressed state, lower nibble = motor_id
    //   0=IDLE  1=RUN  2=STOP  3=FAULT_NOW  4=FAULT_OVER  5=other
    MCI_State_t st = MC_GetSTMStateMotor1();
    uint8_t state_nibble;
    switch (st) {
        case IDLE:             state_nibble = 0; break;
        case RUN:              state_nibble = 1; break;
        case STOP:             state_nibble = 2; break;
        case FAULT_NOW:        state_nibble = 3; break;
        case FAULT_OVER:       state_nibble = 4; break;
        default:               state_nibble = 5; break;
    }
    data[0] = (state_nibble << 4) | (fb->motor_id & 0x0F);
    data[1] = fb->error_code;

    data[2] = (p_int >> 8) & 0xFF;
    data[3] = p_int & 0xFF;

    data[4] = (v_int >> 8) & 0xFF;
    data[5] = v_int & 0xFF;

    data[6] = (t_int >> 8) & 0xFF;
    data[7] = t_int & 0xFF;
}


void CAN_Interrupt_Init(void)
  {
      FDCAN_FilterTypeDef filterConfig;

      // 1. Configure the FDCAN filter profile to ACCEPT EVERYTHING for debugging
      filterConfig.IdType = FDCAN_STANDARD_ID;
      filterConfig.FilterIndex = 0;
      filterConfig.FilterType = FDCAN_FILTER_RANGE;
      filterConfig.FilterConfig = FDCAN_FILTER_TO_RXFIFO0;
      filterConfig.FilterID1 = 0x000;
      filterConfig.FilterID2 = 0x7FF;

      if (HAL_FDCAN_ConfigFilter(&hfdcan1, &filterConfig) != HAL_OK)
      {
          Error_Handler();
      }

      // 2. Start the FDCAN Module Hardware
      if (HAL_FDCAN_Start(&hfdcan1) != HAL_OK)
      {
          Error_Handler();
      }

      // 3. Enable Interrupt Notifications for incoming messages on FIFO 0
      if (HAL_FDCAN_ActivateNotification(&hfdcan1, FDCAN_IT_RX_FIFO0_NEW_MESSAGE, 0) != HAL_OK)
      {
          Error_Handler();
      }
  }

void HAL_FDCAN_RxFifo0Callback(FDCAN_HandleTypeDef *hfdcan, uint32_t RxFifo0ITs)
{
    uint8_t addressed_frame = 0U;

    // Ensure the interrupt source was a new message arriving
    if ((RxFifo0ITs & FDCAN_IT_RX_FIFO0_NEW_MESSAGE) != RESET)
    {
        FDCAN_RxHeaderTypeDef rxHeader;
        uint8_t rxData[8];

        // 1. Fetch message from FDCAN hardware FIFO 0
        if (HAL_FDCAN_GetRxMessage(hfdcan, FDCAN_RX_FIFO0, &rxHeader, rxData) == HAL_OK)
        {
            can_rx_count++;


            // -----------------------------------------

            // 2. Verify incoming identity envelope
            if (rxHeader.Identifier == g_motor_id)
            {
                addressed_frame = 1U;
                /* Auto-recover from a latched fault:
                 * FAULT_OVER means the fault condition has cleared but the
                 * state machine is waiting for an explicit acknowledgment.
                 * Zero the torque target first so the hook doesn't replay a
                 * large command the instant the motor restarts. */
                if (MC_GetSTMStateMotor1() == FAULT_OVER)
                {
                    ImpedanceCtrl.torque_target   = 0.0f;
                    ImpedanceCtrl.position_target = ImpedanceCtrl.position_curr;
                    ImpedanceCtrl.velocity_target = 0.0f;
                    ImpedanceCtrl.kp = 0.0f;
                    ImpedanceCtrl.kd = 0.0f;
                    MC_AcknowledgeFaultMotor1();
                }

                // Unpack payload data bytes
                CAN_Unpack_Command(rxData, &IncomingCmd);

                /* --- INTERPOLATION SETUP --- */
                __disable_irq();
                ImpedanceCtrl.position_target = IncomingCmd.position_target;
                ImpedanceCtrl.velocity_target = IncomingCmd.velocity_target;
                ImpedanceCtrl.torque_target   = IncomingCmd.torque_ff;
                ImpedanceCtrl.kp              = IncomingCmd.kp;
                ImpedanceCtrl.kd              = IncomingCmd.kd;

                ImpedanceCtrl.steps_per_can_frame = 20;
                ImpedanceCtrl.position_step = (ImpedanceCtrl.position_target - ImpedanceCtrl.position_curr) / 20.0f;
                ImpedanceCtrl.velocity_step = (ImpedanceCtrl.velocity_target - ImpedanceCtrl.velocity_curr) / 20.0f;
                ImpedanceCtrl.torque_step   = (ImpedanceCtrl.torque_target   - ImpedanceCtrl.torque_curr)   / 20.0f;
                ImpedanceCtrl.interpolation_counter = 0;
                __enable_irq();
            }

            /* --- STATE MACHINE CONTROL FRAME (CTRL_ID = 0x201) --- */
            else if (rxHeader.Identifier == g_ctrl_id && rxHeader.RxFrameType == FDCAN_DATA_FRAME
                     && rxHeader.DataLength == FDCAN_DLC_BYTES_1)
            {
                addressed_frame = 1U;
                if (rxData[0] == CMD_MOTOR_START)
                {
                    /* Acknowledge any latched fault before starting */
                    MC_AcknowledgeFaultMotor1();
                    /* Zero impedance targets so the hook starts from rest */
                    ImpedanceCtrl.torque_target   = 0.0f;
                    ImpedanceCtrl.position_target = ImpedanceCtrl.position_curr;
                    ImpedanceCtrl.velocity_target = 0.0f;
                    ImpedanceCtrl.kp = 0.0f;
                    ImpedanceCtrl.kd = 0.0f;
                    MC_StartMotor1();
                }
                else if (rxData[0] == CMD_MOTOR_STOP)
                {
                    MC_StopMotor1();
                }
                else if (rxData[0] == CMD_FORCE_ALIGN)
                {
                    /* Request realignment from main loop (can't block in ISR) */
                    force_align = 1;
                }
                else if (rxData[0] == CMD_FEEDBACK_REQUEST)
                {
                    /* Feedback is packed below without changing motor state. */
                }
                else
                {
                    addressed_frame = 0U;
                }
            }



            if (addressed_frame != 0U)
            {
                FDCAN_TxHeaderTypeDef txHeader;
                uint8_t txData[8];

                CurrentState.motor_id = g_motor_id;
                /* Use the sticky "occurred" register so the fault is visible even
                 * after the condition clears and the state has moved to FAULT_OVER.
                 * This register is only reset when MC_AcknowledgeFaultMotor1() is
                 * called (which happens automatically above on the next command). */
                CurrentState.error_code = (uint8_t)(MC_GetOccurredFaultsMotor1() & 0xFF);

                // 1. Fetch Angle: convert s16degrees → radians (-32768..+32767 = -π..+π)
                CurrentState.position_actual = (float)ENCODER_M1._Super.hMecAngle * ((float)M_PI / 32768.0f);

                // 2. Fetch speed (Returns RPM, convert to Rad/s)
                {
                    float speed_rpm = MC_GetAverageMecSpeedMotor1_F();
                    CurrentState.velocity_actual = speed_rpm * RPM_TO_RADS;
                }

                // 3. Fetch Torque (Returns Iq in Amps, convert to Nm)
                {
                    qd_f_t current_qd = MC_GetIqdMotor1_F();
                    CurrentState.torque_actual = current_qd.q * MOTOR_KT;
                }

                CAN_Pack_Feedback(txData, &CurrentState);

                // Configure modern FDCAN transmission settings
                // We use standard response ID: MY_MOTOR_ID + 0x100 (e.g. 0x101)
                txHeader.Identifier = g_motor_id + 0x100;
                txHeader.IdType = FDCAN_STANDARD_ID;
                txHeader.TxFrameType = FDCAN_DATA_FRAME;
                txHeader.DataLength = FDCAN_DLC_BYTES_8;
                txHeader.ErrorStateIndicator = FDCAN_ESI_ACTIVE;
                txHeader.BitRateSwitch = FDCAN_BRS_OFF;
                txHeader.FDFormat = FDCAN_CLASSIC_CAN;
                txHeader.TxEventFifoControl = FDCAN_NO_TX_EVENTS;
                txHeader.MessageMarker = 0;

                // Fire payload back only for frames addressed to this node.
                HAL_FDCAN_AddMessageToTxFifoQ(hfdcan, &txHeader, txData);
            }
        }
    }
}


int _write(int fd, char* ptr, int len) {
  HAL_StatusTypeDef hstatus;

  if (fd == 1 || fd == 2) {
    hstatus = HAL_UART_Transmit(&huart1, (uint8_t *) ptr, len, 100);
    if (hstatus == HAL_OK)
      return len;
    else
      return -1;
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
