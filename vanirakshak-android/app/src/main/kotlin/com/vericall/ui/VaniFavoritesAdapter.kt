package com.vericall.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.vericall.R
import com.vericall.model.VaniContact

class VaniFavoritesAdapter(
    private var favorites: List<VaniContact> = emptyList(),
    private val onContactClicked: (VaniContact) -> Unit,
    private val onCallClicked: (VaniContact) -> Unit
) : RecyclerView.Adapter<VaniFavoritesAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val tvAvatar: TextView = view.findViewById(R.id.tvFavAvatar)
        val tvName: TextView = view.findViewById(R.id.tvFavName)
        val tvPhone: TextView = view.findViewById(R.id.tvFavPhone)
        val tvRelation: TextView = view.findViewById(R.id.tvFavRelation)
        val btnCall: ImageView = view.findViewById(R.id.btnFavQuickCall)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val v = LayoutInflater.from(parent.context).inflate(R.layout.item_favorite_contact, parent, false)
        return ViewHolder(v)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val contact = favorites[position]
        holder.tvName.text = contact.name
        holder.tvPhone.text = contact.phone
        holder.tvRelation.text = contact.relation.ifBlank { if (contact.isVoiceEnrolled) "Voice Enrolled" else "Contact" }

        val initial = contact.name.firstOrNull { it.isLetter() }?.uppercaseChar() ?: '?'
        holder.tvAvatar.text = initial.toString()

        holder.itemView.setOnClickListener { onContactClicked(contact) }
        holder.btnCall.setOnClickListener { onCallClicked(contact) }
    }

    override fun getItemCount(): Int = favorites.size

    fun updateData(newFavorites: List<VaniContact>) {
        favorites = newFavorites
        notifyDataSetChanged()
    }
}
