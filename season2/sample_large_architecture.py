'''
Full network architecture in keras
Includes three models: Transformer, GRU and TCNN
May take a long time to train, but this is the kind of architecture I am thinking about using. 
On top of those three, I might add another transformer encoder model with slightly different configurations. 
'''
def transformer_block(inputs, node, drop_rate, activation):
    attn_output = MultiHeadAttention(num_heads = 4, key_dim = node)(inputs, inputs)
    attn_output = Dropout(drop_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn_output = Dense(node, activation = activation)(out1)
    ffn_output = Dense(node)(ffn_output)
    ffn_output = Dropout(drop_rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2

def transformer_model(inputs, node = 64, activation = 'relu', drop_rate = 0.2, num_layers = 3):
    time_embedding = Time2Vector(N)
    bn = BatchNormalization()(inputs)
    x = time_embedding(bn)
    x = Concatenate()([bn, x])
    x = Conv1D(node*2, 5, activation = activation)(x)
    x = MaxPooling1D(3)(x)
    x = Dropout(drop_rate)(x)
    x = Conv1D(node, 5, activation = activation)(x)
    x = MaxPooling1D(3)(x)
    x = Dropout(drop_rate)(x)
    positions = tf.range(start=0, limit=x.shape[1], delta=1)
    positions = Embedding(input_dim = x.shape[1], output_dim = node)(positions)
    x = x + positions
    for i in range(num_layers):
        x = transformer_block(x, node, drop_rate, activation)
    x = GlobalMaxPooling1D()(x)
    return x

def stacked_gru(inputs, drop_rate = 0.2):
    time_embedding = Time2Vector(N)
    bn = BatchNormalization()(inputs)
    x = time_embedding(bn)
    x = Concatenate()([bn, x])
    x = GRU(128, return_sequences = True)(x)
    x = Dropout(drop_rate)(x)
    x = GRU(128, return_sequences = True)(x)
    x = Dropout(drop_rate)(x)
    x = GRU(128, return_sequences = False)(x)
    x = Dense(64, activation = 'relu')(x)
    return x


def temporal_cnn(inputs, drop_rate = 0.2):
    time_embedding = Time2Vector(N)
    bn = BatchNormalization()(inputs)
    x = time_embedding(bn)
    x = Concatenate()([bn, x])
    x = TCN(nb_filters = 128, dilations = (1,2,4,8,16), dropout_rate = drop_rate, return_sequences = False)(x)
    x = Dense(64, activation = 'relu')(x)
    return x



def build_model():
    price_inputs = Input((N, features_price))
    volume_inputs = Input((N, features_volume))
    # transformer channel
    x_p = transformer_model(price_inputs)
    x_v = transformer_model(volume_inputs)
    merge = Concatenate()([x_p, x_v])
    blend = Dense(64, activation = 'relu')(merge)

    # gru channel
    x_p_gru = stacked_gru(price_inputs)
    x_v_gru = stacked_gru(volume_inputs)
    merge_gru = Concatenate()([x_p_gru, x_v_gru])
    blend_gru = Dense(64, activation = 'relu')(merge_gru)

    # temporal CNN channel
    x_p_tcnn = temporal_cnn(price_inputs)
    x_v_tcnn = temporal_cnn(volume_inputs)
    merge_tcnn = Concatenate()([x_p_tcnn, x_v_tcnn])
    blend_tcnn = Dense(64, activation = 'relu')(merge_tcnn)

    # concatenate outputs
    large_merge = Concatenate()([blend, blend_gru, blend_tcnn])
    large_blend = Dense(64, activation = 'relu')(large_merge)

    large_blend = BatchNormalization()(large_blend)
    outputs = Dense(1, activation = 'relu')(large_blend)
    model = Model(inputs=[price_inputs, volume_inputs], outputs=outputs)
    model.compile(loss = 'mape', optimizer = 'adam', metrics = ['mae','mse','mape'])
    return model
