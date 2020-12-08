import tensorflow as tf
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate, Add, Flatten

def MultiplicativeAttention(hidden_states, units=300):
    """
    Many-to-one attention mechanism.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, attention_shape_out2)
    """
    
    attention_shape_out2 = units
    hidden_size = int(hidden_states.shape[2])
    timesteps = int(hidden_states.shape[1])


    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    H_i = Lambda(lambda x: x[:, :-1, :], output_shape=(timesteps-1,hidden_size), name='source_hidden_states')(hidden_states)

    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(H_i)
    score = dot([score_first_part, h_t], axes =(2, 1), name='attention_score')
    
    attention_weights = Activation('softmax', name='attention_weight')(score)
    
    context_vector = dot([H_i, attention_weights], axes =(1, 1), name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    # attention_vector = Dense(attention_shape_out2, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    attention_vector = Dense(attention_shape_out2, use_bias=True, activation='tanh', name='attention_vector')(pre_activation)

    return attention_vector

def AdditiveAttention(hidden_states,units = 300):
    """
    Many-to-one attention mechanism.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, attention_shape_out2)
    """
    attention_shape_out2 = units
    hidden_size = int(hidden_states.shape[2])
    timesteps = int(hidden_states.shape[1])


    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    H_i = Lambda(lambda x: x[:, :-1, :], output_shape=(timesteps-1,hidden_size), name='source_hidden_states')(hidden_states)    

    score_2_1  = Dense(hidden_size, use_bias=True)(h_t)
    score_2_2 = Dense(hidden_size, use_bias=True)(H_i)
    score_2 = Add(name= 'attention_score_vec')([score_2_1,score_2_2])
    score = Dense(1,use_bias=True,activation='tanh', name = 'attention_score')(score_2)
    score = Flatten()(score)

    attention_weights = Activation('softmax', name='attention_weight')(score)

    context_vector = dot([H_i, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(attention_shape_out2, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    
    return attention_vector

def TemporalAttention(hidden_states):
    """
    Many-to-one attention mechanism.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, attention_shape_out2)
    """
    
    hidden_size = int(hidden_states.shape[2])
    timesteps = int(hidden_states.shape[1])


    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    H_i = Lambda(lambda x: x[:, :-1, :], output_shape=(timesteps-1,hidden_size), name='source_hidden_states')(hidden_states)

    # score_first_part = Dense(hidden_size, use_bias=True, name='attention_score_vec')(H_i)

    score_first_part = Dense(timesteps-1, use_bias=False, name='attention_score_vec')(h_t)

    score = dot([H_i, score_first_part], axes =(1, 1), name='attention_score')
    
    attention_weights = Activation('sigmoid', name='attention_weight')(score)
    
    context_vector = dot([H_i, attention_weights], axes =(2, 1), name='context_vector')

    pre_activation_1 = Dense(hidden_size,use_bias=False, name='pre_activation_1')(h_t) 
    pre_activation_2 = Dense(hidden_size,use_bias=False, name='pre_activation_2')(context_vector)

    attention_vector = Add(name = 'attention_vector')([pre_activation_1,pre_activation_2])
 
    return attention_vector

