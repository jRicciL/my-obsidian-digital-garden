# frozen_string_literal: true

# Turns ==something== in Markdown to <mark>something</mark> in output HTML

Jekyll::Hooks.register [:notes], :pre_render do |note|
    ligatures_format(note)
  end
  
  def ligatures_format(note)
    # Format left arrows
    note.content.gsub!(
      /\B==>/, 
      "⟹"
    )
    note.content.gsub!(
      /\B-->/, 
      "⟶"
    ) 
    note.content.gsub!(
      /\B=>/, 
      "⇒"
    )
    note.content.gsub!(
      /\B->/, 
      "→"
    ) 
    
    # Format right arrows
    note.content.gsub!(
      /\B<==/, 
      "⟸"
    )
    note.content.gsub!(
      /\B<--/, 
      "⟵"
    )
    note.content.gsub!(
      /\B<=/, 
      "⇐"
    )
    note.content.gsub!(
      /\B<-/, 
      "←"
    )
    
  end
  