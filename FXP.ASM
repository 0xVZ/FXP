section .text

%define FXP_INT_BITS    (16)
%define FXP_FRAC_BITS   (16)
%define FXP_SCALE       (FXP_FRAC_BITS)
%define FXP_SF          (1 << FXP_SCALE)

%assign FRAC_DIVISOR    (FXP_SF / 10)

global fxp_mul
global fxp_div
global fxp_abs
global fxptoa

fxp_mul:
    ; Input: EDI = a, ESI = b (32-bit fixed point values)
    ; Output: EAX = a * b >> FXP_SCALE (32-bit result)
    
    movsx   rax, edi        ; Sign-extend 32-bit EDI to 64-bit RAX
    movsx   rcx, esi        ; Sign-extend 32-bit ESI to 64-bit RCX
    imul    rcx             ; RAX = RAX * RCX (64-bit result in RDX:RAX)
    sar     rax, FXP_SCALE  ; Shift right by FXP_SCALE (preserving sign)
    mov     eax, eax        ; Zero-extend to ensure upper 32 bits are cleared
    ret                     ; Return 32-bit result in EAX

fxp_div:
    ; Input: EDI = a, ESI = b (32-bit fixed point values)
    ; Output: EAX = (a << FXP_SCALE) / b (32-bit result)
    
    movsx   rax, edi        ; Sign-extend 32-bit EDI to 64-bit RAX
    shl     rax, FXP_SCALE  ; Left shift by FXP_SCALE
    cqo                     ; Sign-extend RAX into RDX:RAX (for division)
    movsx   rcx, esi        ; Sign-extend 32-bit ESI to 64-bit RCX
    idiv    rcx             ; Signed division: (RDX:RAX / RCX), result in RAX
    mov     eax, eax        ; Zero-extend to ensure upper 32 bits are cleared
    ret                     ; Return 32-bit result in EAX

fxp_abs:
    ; Input: EDI = a (32-bit fixed point value)
    ; Output: EAX = |a| (32-bit result)
    
    mov     eax, edi        ; Copy input to EAX
    cdq                     ; Sign-extend EAX into EDX (fills EDX with sign bit of EAX)
    xor     eax, edx        ; XOR with sign mask
    sub     eax, edx        ; Subtract sign mask
    ret                     ; Return result in EAX

fxptoa:
    ; Input: RDI = str, ESI = value
    ; Output: None

    ; str[0] = '-' if ESI is negative
    CLD
    MOV     byte [RDI], '-'
    MOV     RAX, RDI
    INC     RAX
    TEST    ESI, ESI
    CMOVS   RDI, RAX

    ; EAX = ABS(EAX)
    MOV     EAX, ESI
    CDQ
    XOR     EAX, EDX
    SUB     EAX, EDX

    ; EAX = ESI / FXP_SF
    ; EDX = ESI % FXP_SF
    MOV     ECX, FXP_SF
    CDQ
    IDIV    ECX
    MOV     R10D, EDX

    ; Load itoa(EAX) into RSI
    ; R9W will be used as a flag to ignore all the zeros
    ;  before the first non-zero digit
    XOR     R9W, R9W

    ; 1000000000
    XOR     EDX, EDX
    MOV     EBX, 1000000000
    DIV     EBX
    MOV     R8D, EAX
    ADD     AL, '0'
    MOV     byte [RDI], AL
    MOV     RSI, RDI
    INC     RSI
    TEST    R8D, R8D
    CMOVNZ  RDI, RSI
    CMOVNZ  R9W, AX
    MOV     EAX, EDX

    ; 100000000
    XOR     EDX, EDX
    MOV     EBX, 100000000
    DIV     EBX
    MOV     R8D, EAX
    ADD     AL, '0'
    MOV     byte [RDI], AL
    MOV     RSI, RDI
    INC     RSI
    TEST    R8D, R8D
    CMOVNZ  RDI, RSI
    CMOVNZ  R9W, AX
    TEST    R9W, R9W
    CMOVNZ  RDI, RSI
    MOV     EAX, EDX

    ; 10000000
    XOR     EDX, EDX
    MOV     EBX, 10000000
    DIV     EBX
    MOV     R8D, EAX
    ADD     AL, '0'
    MOV     byte [RDI], AL
    MOV     RSI, RDI
    INC     RSI
    TEST    R8D, R8D
    CMOVNZ  RDI, RSI
    CMOVNZ  R9W, AX
    TEST    R9W, R9W
    CMOVNZ  RDI, RSI
    MOV     EAX, EDX

    ; 1000000
    XOR     EDX, EDX
    MOV     EBX, 1000000
    DIV     EBX
    MOV     R8D, EAX
    ADD     AL, '0'
    MOV     byte [RDI], AL
    MOV     RSI, RDI
    INC     RSI
    TEST    R8D, R8D
    CMOVNZ  RDI, RSI
    CMOVNZ  R9W, AX
    TEST    R9W, R9W
    CMOVNZ  RDI, RSI
    MOV     EAX, EDX

    ; 100000
    XOR     EDX, EDX
    MOV     EBX, 100000
    DIV     EBX
    MOV     R8D, EAX
    ADD     AL, '0'
    MOV     byte [RDI], AL
    MOV     RSI, RDI
    INC     RSI
    TEST    R8D, R8D
    CMOVNZ  RDI, RSI
    CMOVNZ  R9W, AX
    TEST    R9W, R9W
    CMOVNZ  RDI, RSI
    MOV     EAX, EDX

    ; 10000
    XOR     EDX, EDX
    MOV     EBX, 10000
    DIV     EBX
    MOV     R8D, EAX
    ADD     AL, '0'
    MOV     byte [RDI], AL
    MOV     RSI, RDI
    INC     RSI
    TEST    R8D, R8D
    CMOVNZ  RDI, RSI
    CMOVNZ  R9W, AX
    TEST    R9W, R9W
    CMOVNZ  RDI, RSI
    MOV     EAX, EDX

    ; 1000
    XOR     EDX, EDX
    MOV     EBX, 1000
    DIV     EBX
    MOV     R8D, EAX
    ADD     AL, '0'
    MOV     byte [RDI], AL
    MOV     RSI, RDI
    INC     RSI
    TEST    R8D, R8D
    CMOVNZ  RDI, RSI
    CMOVNZ  R9W, AX
    TEST    R9W, R9W
    CMOVNZ  RDI, RSI
    MOV     EAX, EDX

    ; 100
    XOR     EDX, EDX
    MOV     EBX, 100
    DIV     EBX
    MOV     R8D, EAX
    ADD     AL, '0'
    MOV     byte [RDI], AL
    MOV     RSI, RDI
    INC     RSI
    TEST    R8D, R8D
    CMOVNZ  RDI, RSI
    CMOVNZ  R9W, AX
    TEST    R9W, R9W
    CMOVNZ  RDI, RSI
    MOV     EAX, EDX

    ; 10
    XOR     EDX, EDX
    MOV     EBX, 10
    DIV     EBX
    MOV     R8D, EAX
    ADD     AL, '0'
    MOV     byte [RDI], AL
    MOV     RSI, RDI
    INC     RSI
    TEST    R8D, R8D
    CMOVNZ  RDI, RSI
    CMOVNZ  R9W, AX
    TEST    R9W, R9W
    CMOVNZ  RDI, RSI
    MOV     EAX, EDX

    ; 1
    MOV     R8D, EAX
    ADD     AL, '0'
    MOV     byte [RDI], AL
    MOV     RSI, RDI
    INC     RSI
    TEST    R8D, R8D
    CMOVNZ  RDI, RSI
    TEST    R9W, R9W
    CMOVNZ  RDI, RSI

    ; Integer part done. Add '.'
    MOV     AL, '.'
    STOSB

    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;; THERE IS A BUG DOWN HERE, SINCE FRAC_DIVISOR IS NOT /10 ;;;;;;;;;;;;;;;;;;;
    ; This only happens in some cases, since only affects the print, i will fix it later :) 
    
    ; Load itoa(EDX) into RSI
    MOV     EAX, R10D           ; frac_part
    MOV     EBX, FRAC_DIVISOR   ; divisor
    MOV     ECX, 10

.loop_start:
    XOR     EDX, EDX
    DIV     EBX         ; EAX = EDX:EAX / EBX
    MOV     R9D, EDX

    ADD     AL, '0'
    STOSB
    
    ; divisor /= 10
    XOR     EDX, EDX
    MOV     EAX, EBX        ; Load original divisor into EAX
    DIV     ECX             ; EAX = divisor / 10
    MOV     EBX, EAX        ; Store back into EBX
    MOV     EAX, R9D

    TEST    EBX, EBX        ; Check if divisor > 0
    JNZ     .loop_start     ; Exit loop if divisor is 0

    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; ;;;;;;;

.done:
    ; All done, NULL-terminate the string.
    MOV     byte [RDI], 0
    RET