"""Experimental K3 visible-EOS mapping; leaves other stop policies unchanged."""
import logging

EOS = 163586
HIDDEN_STOP = 163585
PAD = 163839


def wrap_stop_mapper(original):
    announced = set()

    def apply(params, conditions):
        visible = set(conditions.get('stop_token_ids_visible') or [])
        hidden = set(conditions.get('stop_token_ids_hidden') or [])
        supported = (
            visible == {EOS}
            and not conditions.get('ignore_eos')
            and hidden in (set(), {HIDDEN_STOP})
            and not conditions.get('stop')
            and not conditions.get('stop_token_ids')
            and not conditions.get('min_tokens')
            and not getattr(params, 'bad', None)
            and not getattr(params, 'min_tokens', None)
            and not getattr(params, 'use_beam_search', False)
            and not getattr(params, 'stop', None)
            and not getattr(params, 'stop_token_ids', None)
            and not getattr(params, 'ignore_eos', False)
        )
        if supported:
            # Use STOP_WORDS instead of END_ID so the result layer can retain
            # the real generated EOS. No synthetic tokens are inserted.
            params.end_id = -1
            if params.pad_id is None:
                params.pad_id = PAD
            # Rust retains the unchanged visibility rules for both tokens.
            # The engine returns real stop tokens; it does not synthesize any.
            params.stop_token_ids = sorted(visible | hidden)
            params.include_stop_str_in_output = True
            params.ignore_eos = False
        else:
            original(params, conditions)
        if supported not in announced:
            logging.warning('K3 visible-EOS experiment: engine_stop=%s conditions=%s',
                            supported, conditions)
            announced.add(supported)
    return apply
