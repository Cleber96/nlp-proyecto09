# Modelos Seq2Seq No Autoregresivos (Non-Autoregressive Translation, NAT)
## Objetivo
Reducir la latencia de generación eliminando la dependencia causal token-a-token, a costa de retos de multimodalidad en la distribución de salidas.
## Descripción
- Los modelos seq2seq clásicos generan cada token condicionándose en todos los anteriores
- En cambio, los NAT introducen una factorización paralela