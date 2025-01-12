from torch import nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_ff = nn.LayerNorm(embed_dim)
    
    def forward(self, query, key, value):
        # query: (batch_size, embed_dim)
        # key, value: (batch_size, embed_dim)
        
        # Reshape for MultiheadAttention: (seq_len, batch_size, embed_dim)
        query = query.unsqueeze(0)  # (1, batch_size, embed_dim)
        key = key.unsqueeze(0)      # (1, batch_size, embed_dim)
        value = value.unsqueeze(0)  # (1, batch_size, embed_dim)
        
        attn_output, _ = self.multihead_attn(query, key, value)  # (1, batch_size, embed_dim)
        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm(attn_output + query)  # Residual connection
        
        # Feed Forward
        ff_output = self.feed_forward(attn_output)  # (1, batch_size, embed_dim)
        ff_output = self.norm_ff(ff_output + attn_output)  # Residual connection
        
        # Remove sequence dimension
        ff_output = ff_output.squeeze(0)  # (batch_size, embed_dim)
        return ff_output