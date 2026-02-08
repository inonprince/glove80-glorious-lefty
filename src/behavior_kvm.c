/*
 * Copyright (c) 2026
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>

#include <zephyr/device.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include <drivers/behavior.h>
#include <dt-bindings/zmk/keys.h>

#include <zmk/behavior.h>
#if !IS_ENABLED(CONFIG_ZMK_SPLIT) || IS_ENABLED(CONFIG_ZMK_SPLIT_ROLE_CENTRAL)
#include <zmk/events/keycode_state_changed.h>
#endif

#if IS_ENABLED(CONFIG_ZMK_RGB_UNDERGLOW)
#include <zmk/events/underglow_color_changed.h>
#endif

LOG_MODULE_DECLARE(zmk, CONFIG_ZMK_LOG_LEVEL);

/*
 * Volatile KVM state:
 * - next_is_two controls the output sequence (1,2,1,2,...)
 * - last_is_two drives the RGB indicator
 */
static bool kvm_next_is_two = false;
static bool kvm_last_is_two = false;

#if !IS_ENABLED(CONFIG_ZMK_SPLIT) || IS_ENABLED(CONFIG_ZMK_SPLIT_ROLE_CENTRAL)
static int tap_encoded(uint32_t encoded, int64_t timestamp) {
    int err = raise_zmk_keycode_state_changed_from_encoded(encoded, true, timestamp);
    if (err) {
        return err;
    }

    return raise_zmk_keycode_state_changed_from_encoded(encoded, false, timestamp);
}

static int send_kvm_sequence(bool send_two, int64_t timestamp) {
    int err = tap_encoded(LCTRL, timestamp);
    if (err) {
        return err;
    }

    err = tap_encoded(LCTRL, timestamp);
    if (err) {
        return err;
    }

    return tap_encoded(send_two ? N2 : N1, timestamp);
}
#endif

static void refresh_kvm_indicator(void) {
#if IS_ENABLED(CONFIG_ZMK_RGB_UNDERGLOW)
    raise_zmk_underglow_color_changed((struct zmk_underglow_color_changed){
        .layers = UINT32_MAX,
        .wakeup = true,
    });
#endif
}

#undef DT_DRV_COMPAT
#define DT_DRV_COMPAT zmk_behavior_kvm_switch

#if DT_HAS_COMPAT_STATUS_OKAY(DT_DRV_COMPAT)

static int on_kvm_switch_pressed(struct zmk_behavior_binding *binding,
                                 struct zmk_behavior_binding_event event) {
    ARG_UNUSED(binding);

    bool send_two = kvm_next_is_two;

#if !IS_ENABLED(CONFIG_ZMK_SPLIT) || IS_ENABLED(CONFIG_ZMK_SPLIT_ROLE_CENTRAL)
    int err = send_kvm_sequence(send_two, event.timestamp);
    if (err) {
        return err;
    }
#else
    ARG_UNUSED(event);
#endif

    kvm_last_is_two = send_two;
    kvm_next_is_two = !kvm_next_is_two;
    refresh_kvm_indicator();

    return ZMK_BEHAVIOR_OPAQUE;
}

static int on_kvm_switch_released(struct zmk_behavior_binding *binding,
                                  struct zmk_behavior_binding_event event) {
    ARG_UNUSED(binding);
    ARG_UNUSED(event);

    return ZMK_BEHAVIOR_OPAQUE;
}

static const struct behavior_driver_api kvm_switch_driver_api = {
    .binding_pressed = on_kvm_switch_pressed,
    .binding_released = on_kvm_switch_released,
    .locality = BEHAVIOR_LOCALITY_GLOBAL,
#if IS_ENABLED(CONFIG_ZMK_BEHAVIOR_METADATA)
    .get_parameter_metadata = zmk_behavior_get_empty_param_metadata,
#endif
};

#define KVM_SWITCH_INST(n)                                                                         \
    BEHAVIOR_DT_INST_DEFINE(n, NULL, NULL, NULL, NULL, POST_KERNEL,                                \
                            CONFIG_KERNEL_INIT_PRIORITY_DEFAULT, &kvm_switch_driver_api);

DT_INST_FOREACH_STATUS_OKAY(KVM_SWITCH_INST)

#endif /* DT_HAS_COMPAT_STATUS_OKAY(DT_DRV_COMPAT) */

#undef DT_DRV_COMPAT
#define DT_DRV_COMPAT zmk_behavior_kvm_state_color

#if DT_HAS_COMPAT_STATUS_OKAY(DT_DRV_COMPAT)

static int on_kvm_state_color_pressed(struct zmk_behavior_binding *binding,
                                      struct zmk_behavior_binding_event event) {
    ARG_UNUSED(binding);
    ARG_UNUSED(event);

    return kvm_last_is_two ? 0xFF0000 : 0;
}

static int on_kvm_state_color_released(struct zmk_behavior_binding *binding,
                                       struct zmk_behavior_binding_event event) {
    ARG_UNUSED(binding);
    ARG_UNUSED(event);

    return ZMK_BEHAVIOR_OPAQUE;
}

static const struct behavior_driver_api kvm_state_color_driver_api = {
    .binding_pressed = on_kvm_state_color_pressed,
    .binding_released = on_kvm_state_color_released,
    .locality = BEHAVIOR_LOCALITY_GLOBAL,
#if IS_ENABLED(CONFIG_ZMK_BEHAVIOR_METADATA)
    .get_parameter_metadata = zmk_behavior_get_empty_param_metadata,
#endif
};

#define KVM_STATE_COLOR_INST(n)                                                                    \
    BEHAVIOR_DT_INST_DEFINE(n, NULL, NULL, NULL, NULL, POST_KERNEL,                                \
                            CONFIG_KERNEL_INIT_PRIORITY_DEFAULT,                                   \
                            &kvm_state_color_driver_api);

DT_INST_FOREACH_STATUS_OKAY(KVM_STATE_COLOR_INST)

#endif /* DT_HAS_COMPAT_STATUS_OKAY(DT_DRV_COMPAT) */
