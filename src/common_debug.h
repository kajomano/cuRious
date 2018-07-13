#ifndef COMMON_DEBUG
#define COMMON_DEBUG

#define DEBUG_PRINTS 1

// Debug print macros
#ifdef DEBUG_PRINTS
#define debugPrint(ans){ (ans); }  // Debug prints on
#else
#define debugPrint(ans){ }         // Debug prints off
#endif

#endif
