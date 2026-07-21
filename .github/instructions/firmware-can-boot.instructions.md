---
description: STM32G4 firmware pitfalls for the jumping_robot boards (CAN bus, boot config, interrupts). Load when editing or debugging any Core/Src/main.c or CAN/FDCAN code in flywheel*, spring_actuator, or Jumping_Sensors.
applyTo: '**/{Core/Src,Src}/main.c'
---

# jumping_robot STM32G4 firmware — CAN & boot learnings

Hard-won debugging knowledge for the STM32G4 boards on this project. The nodes
share a 1 Mbit/s classic-CAN bus decoded by `motor_control_gui.py`:
flywheel_x `0x01`, flywheel_y `0x02`, spring_act `0x03`, imu_sensor `0x04`.
Feedback ID = command ID + `0x100` (e.g. `0x104`). Motor boards are
STSPIN32G4 (embedded STM32G431); the sensor board is an STM32G4A1KEUx.

## Learnings

- **Never transmit CAN feedback unconditionally in the RX callback.** Each board
  must reply **only** to frames addressed to it — gate the TX with
  `if (rxHeader.Identifier == g_motor_id || rxHeader.Identifier == g_ctrl_id)`.
  If the feedback TX sits outside this check, every board replies to every frame
  (including other boards' feedback), producing an exponential CAN storm that
  kills the whole bus the instant more than one node is connected. It works fine
  with a single board (the USB-CAN adapter never sends feedback), which makes the
  bug invisible until full assembly. This bit both `flywheel2` and
  `spring_actuator`.

- **Set `SCB->VTOR` explicitly at the top of `main()`** (`USER CODE BEGIN 1`):
  `SCB->VTOR = FLASH_BASE; __DSB();`. CubeMX leaves `USER_VECT_TAB_ADDRESS`
  commented out in `system_stm32g4xx.c`, so `SystemInit()` never writes VTOR and
  it stays at its reset value `0x00000000`. On boards where address 0 is not
  aliased to app flash, every interrupt (SysTick, FDCAN, ...) vectors into the
  wrong table and silently never runs — `HAL_GetTick()` stays frozen at 0, so any
  `HAL_GetTick()`-gated CAN send never fires and the loop reports 0 Hz, even
  though polled code (UART, SPI IMU reads) keeps running. Diagnose by printing
  `SCB->VTOR` (should be `0x08000000`) and a counter incremented inside
  `SysTick_Handler`.

- **Sensor board (STM32G4A1KEUx, UFQFPN32) has no dedicated BOOT0 pin — it is
  shared as `PB8-BOOT0`.** If PB8 floats, the chip may boot the system bootloader
  on a cold power-up (no UART, no app), while an ST-Link/debugger reset always
  forces a flash boot and hides the problem. Fix via option bytes in
  STM32CubeProgrammer: **nSWBOOT0 = 0 (unchecked)** and **nBOOT0 = 1 (checked)**
  → always boot from main flash regardless of PB8. (Alternative: 10 kΩ pulldown
  PB8→GND.) The STSPIN32G4 driver boards boot from flash correctly on cold start
  (LEDs blink), so they don't need this.

- **FDCAN on this project runs classic CAN at 1 Mbit with
  `AutoRetransmission = DISABLE` and no bus-off recovery.** `TEC=0` therefore does
  NOT prove a frame was ACKed (an un-ACKed one-shot frame is just dropped). If a
  board ever needs to survive transient bus-off, add explicit recovery; today the
  design relies on the periodic re-send.

- **FDCAN RX filter must route to FIFO0 on boards that read FIFO0.** The sensor
  originally passed `FDCAN_REJECT_REMOTE` (value `0x01` = `ACCEPT_IN_RX_FIFO1`) in
  `HAL_FDCAN_ConfigGlobalFilter`, sending all non-matching frames to FIFO1 which
  it never reads. Use an accept-all range filter into FIFO0. The motor boards work
  because they don't call `ConfigGlobalFilter` (HW default = FIFO0).

- **Sensor FDCAN clock = PCLK1 = 16 MHz (HSI, PLL off), timing 2/6/1 → 1 Mbit.**
  Motor boards use 80 MHz PCLK1, timing 5/12/3 → 1 Mbit. Bitrates match — do NOT
  "fix" the sensor's prescalers to match the motors' numbers.

- **TEC/REC live in `FDCAN_ErrorCountersTypeDef` via
  `HAL_FDCAN_GetErrorCounters()`**, NOT in `FDCAN_ProtocolStatusTypeDef` (which
  has `LastErrorCode`, `BusOff`, `ErrorPassive`). Mixing these up is a compile
  error (`'FDCAN_ProtocolStatusTypeDef' has no member named 'TxErrorCnt'`).

- **The GUI only commands the selected node**, so only that node streams feedback
  at a time — this is by design, not a bus fault. Seeing just one node respond
  after full assembly is expected unless the GUI is changed to poll all nodes.

- **Keep all edits inside `USER CODE BEGIN/END` blocks** so CubeMX regeneration
  from the `.ioc` doesn't wipe them (the VTOR write, CAN helpers, RX callback,
  loop logic).
